export const runtime = "nodejs"; // ensure Node runtime (needed for formdata)

export async function POST(req: Request) {
  try {
    const backend = process.env.BACKEND_URL!;
    const form = await req.formData();          // contains the "file"
    const res = await fetch(`${backend}/predict`, {
      method: "POST",
      body: form,                                // forward as-is
    });

    // pass through response + content-type
    const text = await res.text();
    return new Response(text, {
      status: res.status,
      headers: {
        "content-type": res.headers.get("content-type") ?? "application/json",
      },
    });
  } catch (e: any) {
    return new Response(
      JSON.stringify({ error: "proxy_failed", message: String(e) }),
      { status: 500 }
    );
  }
}
