"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/db";
import {
  hashPassword,
  verifyPassword,
  signToken,
  AUTH_COOKIE,
  cookieOptions,
} from "@/lib/auth";

// ─── Register ─────────────────────────────────────────────────────────────────

export type AuthState = { error?: string } | null;

export async function register(_prevState: AuthState, formData: FormData): Promise<AuthState> {
  const name = (formData.get("name") as string)?.trim();
  const email = (formData.get("email") as string)?.trim().toLowerCase();
  const password = formData.get("password") as string;
  const confirmPassword = formData.get("confirmPassword") as string;

  // Basic validation
  if (!email || !password) {
    return { error: "Email and password are required." };
  }
  if (password.length < 8) {
    return { error: "Password must be at least 8 characters." };
  }
  if (password !== confirmPassword) {
    return { error: "Passwords do not match." };
  }

  // Check for existing user
  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) {
    return { error: "An account with this email already exists." };
  }

  // Create user
  const passwordHash = await hashPassword(password);
  const user = await prisma.user.create({
    data: { email, name: name || null, passwordHash },
  });

  // Sign JWT and set cookie
  const token = signToken({ userId: user.id, email: user.email, name: user.name });
  const cookieStore = await cookies();
  cookieStore.set(AUTH_COOKIE, token, cookieOptions);

  redirect("/");
}

// ─── Login ────────────────────────────────────────────────────────────────────

export async function login(_prevState: AuthState, formData: FormData): Promise<AuthState> {
  const email = (formData.get("email") as string)?.trim().toLowerCase();
  const password = formData.get("password") as string;

  if (!email || !password) {
    return { error: "Email and password are required." };
  }

  const user = await prisma.user.findUnique({ where: { email } });
  if (!user) {
    return { error: "Invalid email or password." };
  }

  const valid = await verifyPassword(password, user.passwordHash);
  if (!valid) {
    return { error: "Invalid email or password." };
  }

  const token = signToken({ userId: user.id, email: user.email, name: user.name });
  const cookieStore = await cookies();
  cookieStore.set(AUTH_COOKIE, token, cookieOptions);

  redirect("/");
}

// ─── Logout ───────────────────────────────────────────────────────────────────

export async function logout() {
  const cookieStore = await cookies();
  cookieStore.delete(AUTH_COOKIE);
  redirect("/login");
}
