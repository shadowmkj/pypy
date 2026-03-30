import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { verifyToken, AUTH_COOKIE } from "@/lib/auth";
import { prisma } from "@/lib/db";

export async function GET() {
  // ── Auth ──────────────────────────────────────────────────────────────────
  const cookieStore = await cookies();
  const token = cookieStore.get(AUTH_COOKIE)?.value;
  const session = token ? verifyToken(token) : null;

  if (!session) {
    return NextResponse.json({ error: "Unauthorised" }, { status: 401 });
  }

  // ── Fetch sessions ────────────────────────────────────────────────────────
  const chatSessions = await prisma.chatSession.findMany({
    where: { userId: session.userId },
    orderBy: { updatedAt: "desc" },
    include: {
      chats: {
        orderBy: { createdAt: "desc" },
        take: 1,
        select: { content: true, isResponse: true, createdAt: true },
      },
    },
  });

  const result = chatSessions.map((s) => ({
    id: s.id,
    title: s.title,
    updatedAt: s.updatedAt,
    lastMessage: s.chats[0]
      ? {
          content: s.chats[0].content.slice(0, 120),
          isResponse: s.chats[0].isResponse,
          createdAt: s.chats[0].createdAt,
        }
      : null,
  }));

  return NextResponse.json(result);
}

export async function DELETE() {
  // ── Auth ──────────────────────────────────────────────────────────────────
  const cookieStore = await cookies();
  const token = cookieStore.get(AUTH_COOKIE)?.value;
  const session = token ? verifyToken(token) : null;

  if (!session) {
    return NextResponse.json({ error: "Unauthorised" }, { status: 401 });
  }

  // ── Get all session ids for this user ──────────────────────────────────────
  const userSessions = await prisma.chatSession.findMany({
    where: { userId: session.userId },
    select: { id: true },
  });

  const sessionIds = userSessions.map((s) => s.id);

  if (sessionIds.length === 0) {
    return NextResponse.json({ success: true, deleted: 0 });
  }

  // ── Delete all messages, then all sessions ─────────────────────────────────
  await prisma.$transaction([
    prisma.chat.deleteMany({ where: { sessionId: { in: sessionIds } } }),
    prisma.chatSession.deleteMany({ where: { id: { in: sessionIds } } }),
  ]);

  return NextResponse.json({ success: true, deleted: sessionIds.length });
}
