import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { verifyToken, AUTH_COOKIE } from "@/lib/auth";
import { prisma } from "@/lib/db";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  // ── Auth ──────────────────────────────────────────────────────────────────
  const cookieStore = await cookies();
  const token = cookieStore.get(AUTH_COOKIE)?.value;
  const authSession = token ? verifyToken(token) : null;

  if (!authSession) {
    return NextResponse.json({ error: "Unauthorised" }, { status: 401 });
  }

  // ── Parse body ────────────────────────────────────────────────────────────
  let message: string;
  let sessionId: number | null = null;

  try {
    const body = await request.json();
    message = (body.message as string)?.trim();
    sessionId = body.sessionId ? Number(body.sessionId) : null;

    if (!message) throw new Error("Empty message");
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  // ── Ensure ChatSession exists ──────────────────────────────────────────────
  let isNewSession = false;

  if (sessionId) {
    // Verify ownership
    const existing = await prisma.chatSession.findUnique({
      where: { id: sessionId },
      select: { userId: true },
    });
    if (!existing || existing.userId !== authSession.userId) {
      // Silently create a new session instead of erroring
      sessionId = null;
    }
  }

  if (!sessionId) {
    const newSession = await prisma.chatSession.create({
      data: {
        userId: authSession.userId,
        title: message.slice(0, 60),
      },
    });
    sessionId = newSession.id;
    isNewSession = true;
  }

  // ── Persist user message ──────────────────────────────────────────────────
  await prisma.chat.create({
    data: {
      sessionId,
      isResponse: false, // false = user message
      content: message,
    },
  });

  // ── Stream from FastAPI backend ───────────────────────────────────────────
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      // First frame: send sessionId so client can track it
      controller.enqueue(
        encoder.encode(
          `event: session\ndata: ${JSON.stringify({ sessionId, isNew: isNewSession })}\n\n`
        )
      );

      let backendResponse: Response;
      try {
        backendResponse = await fetch(`${BACKEND_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message }),
          signal: request.signal,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Backend unreachable";
        controller.enqueue(encoder.encode(`data: [ERROR]${msg}\n\n`));
        controller.close();
        return;
      }

      if (!backendResponse.ok) {
        controller.enqueue(
          encoder.encode(`data: [ERROR]Backend error: ${backendResponse.statusText}\n\n`)
        );
        controller.close();
        return;
      }

      const reader = backendResponse.body?.getReader();
      if (!reader) {
        controller.enqueue(encoder.encode(`data: [ERROR]No response body\n\n`));
        controller.close();
        return;
      }

      const decoder = new TextDecoder("utf-8");
      let fullResponse = "";
      let buffer = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          let lineEnd = buffer.indexOf("\n\n");
          while (lineEnd !== -1) {
            const eventStr = buffer.slice(0, lineEnd).trim();
            buffer = buffer.slice(lineEnd + 2);

            if (eventStr.startsWith("data: ")) {
              const data = eventStr.slice(6);

              if (data === "[DONE]") {
                break;
              }
              if (data.startsWith("[ERROR]")) {
                controller.enqueue(encoder.encode(`data: ${data}\n\n`));
                controller.close();
                return;
              }

              // Accumulate and forward the chunk
              try {
                const parsed = JSON.parse(data);
                fullResponse += parsed;
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(parsed)}\n\n`)
                );
              } catch {
                fullResponse += data;
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
              }
            }

            lineEnd = buffer.indexOf("\n\n");
          }
        }
      } catch (err) {
        if (!(err instanceof Error && err.name === "AbortError")) {
          console.error("[/api/chat/send] Stream error:", err);
        }
      } finally {
        reader.releaseLock();
      }

      // ── Persist AI response & update session timestamp ────────────────────
      if (fullResponse) {
        await prisma.chat.create({
          data: {
            sessionId: sessionId!,
            isResponse: true, // true = AI response
            content: fullResponse,
          },
        });

        await prisma.chatSession.update({
          where: { id: sessionId! },
          data: { updatedAt: new Date() },
        });
      }

      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  return new NextResponse(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
