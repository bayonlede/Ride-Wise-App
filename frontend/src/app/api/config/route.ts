import { NextResponse } from "next/server";

/**
 * Returns the backend API URL at runtime.
 * Railway injects variables at runtime, so this works even when
 * NEXT_PUBLIC_* wasn't available at build time.
 */
export async function GET() {
  const apiUrl =
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.API_URL ||
    "http://localhost:8000";
  return NextResponse.json({ apiUrl: apiUrl.replace(/\/$/, "") });
}
