const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";
const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade"
]);

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 300;

type RouteContext = {
  params: Promise<{
    path?: string[];
  }>;
};

function buildUpstreamHeaders(headers: Headers): Headers {
  const upstreamHeaders = new Headers();
  headers.forEach((value, key) => {
    if (!HOP_BY_HOP_HEADERS.has(key.toLowerCase())) {
      upstreamHeaders.set(key, value);
    }
  });
  return upstreamHeaders;
}

function buildDownstreamHeaders(headers: Headers): Headers {
  const downstreamHeaders = new Headers();
  headers.forEach((value, key) => {
    if (!HOP_BY_HOP_HEADERS.has(key.toLowerCase())) {
      downstreamHeaders.set(key, value);
    }
  });
  return downstreamHeaders;
}

async function proxy(request: Request, context: RouteContext): Promise<Response> {
  const { path = [] } = await context.params;
  const pathname = path.map(encodeURIComponent).join("/");
  const target = new URL(`${FASTAPI_BASE_URL.replace(/\/$/, "")}/${pathname}${new URL(request.url).search}`);
  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  const body = hasBody ? await request.arrayBuffer() : undefined;

  const upstream = await fetch(target, {
    method: request.method,
    headers: buildUpstreamHeaders(request.headers),
    body,
    cache: "no-store",
    signal: AbortSignal.timeout(300_000),
  });

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: buildDownstreamHeaders(upstream.headers),
  });
}

export async function GET(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context);
}

export async function POST(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context);
}

export async function PUT(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context);
}

export async function PATCH(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context);
}

export async function DELETE(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context);
}

export async function OPTIONS(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context);
}

export async function HEAD(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context);
}
