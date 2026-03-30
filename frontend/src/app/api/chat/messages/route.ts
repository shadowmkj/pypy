import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { verifyToken, AUTH_COOKIE } from "@/lib/auth";
import { prisma } from "@/lib/db";

export async function GET(request: NextRequest) {
  // ── Auth ──────────────────────────────────────────────────────────────────
  const cookieStore = await cookies();
  const token = cookieStore.get(AUTH_COOKIE)?.value;
  const session = token ? verifyToken(token) : null;

  if (!session) {
    return NextResponse.json({ error: "Unauthorised" }, { status: 401 });
  }

  // ── Parse sessionId ───────────────────────────────────────────────────────
  const { searchParams } = new URL(request.url);
  const rawId = searchParams.get("sessionId");
  const sessionId = rawId ? parseInt(rawId, 10) : NaN;

  if (isNaN(sessionId)) {
    return NextResponse.json({ error: "sessionId is required" }, { status: 400 });
  }

  // ── Verify ownership ──────────────────────────────────────────────────────
  const chatSession = await prisma.chatSession.findUnique({
    where: { id: sessionId },
    select: { userId: true },
  });

  if (!chatSession || chatSession.userId !== session.userId) {
    return NextResponse.json({ error: "Session not found" }, { status: 404 });
  }

  // ── Fetch messages ────────────────────────────────────────────────────────
  const chats = await prisma.chat.findMany({
    where: { sessionId },
    orderBy: { createdAt: "asc" },
    select: { id: true, isResponse: true, content: true, createdAt: true },
  });

  return NextResponse.json(chats);
}
