import { type NextRequest, NextResponse } from "next/server"

const REMOTE_BASE = "https://sandbox.minepi.com/api"

/**
 * Pass-through proxy for every MinePI REST call.
 *  – All query-strings, headers and HTTP methods are preserved.
 *  – Runs on the server so no CORS pre-flight reaches the browser.
 */
async function proxy(
  request: NextRequest,
  {
    params,
  }: {
    params: { path: string[] }
  },
) {
  /* Build the remote URL */
  const remoteUrl = `${REMOTE_BASE}/${params.path.join("/")}${request.nextUrl.search}`

  /* Clone headers, but drop Host / Accept-Encoding etc. */
  const headers: HeadersInit = {}
  request.headers.forEach((value, key) => {
    if (!["host", "connection", "accept-encoding"].includes(key)) {
      headers[key] = value
    }
  })

  /* Forward */
  const init: RequestInit = {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
    cache: "no-store",
  }

  const res = await fetch(remoteUrl, init)

  /* Pipe response back to the browser */
  const body = await res.arrayBuffer()
  const responseHeaders = new Headers(res.headers)
  return new NextResponse(body, { status: res.status, headers: responseHeaders })
}

/* Match every HTTP method */
export const GET = proxy
export const POST = proxy
export const PUT = proxy
export const PATCH = proxy
export const DELETE = proxy
