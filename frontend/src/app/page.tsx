import { cookies } from "next/headers";
import { verifyToken, AUTH_COOKIE } from "@/lib/auth";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { redirect } from "next/navigation";

export default async function Home() {
  const cookieStore = await cookies();
  const token = cookieStore.get(AUTH_COOKIE)?.value;
  const session = token ? verifyToken(token) : null;

  if (!session) {
    redirect("/login");
  }

  const userInitials = session.name
    ? session.name
        .split(" ")
        .map((n: string) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2)
    : session.email[0].toUpperCase();

  const userName = session.name ?? session.email;

  return (
    <main className="flex min-h-screen flex-col items-center justify-between">
      <div className="w-full max-w-7xl mx-auto h-screen relative">
        <ChatContainer userInitials={userInitials} userName={userName} />
      </div>
    </main>
  );
}
