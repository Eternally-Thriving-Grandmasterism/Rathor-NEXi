// app/api/grok/route.js
// Rathor-NEXi Sovereign Grok Proxy – Edge + CORS + Streaming
// MIT License – Autonomicity Games Inc. 2026

import { NextResponse } from 'next/server';

export const runtime = 'edge';

export async function OPTIONS() {
  return new NextResponse(null, {
    headers: corsHeaders(),
  });
}

export async function POST(request) {
  try {
    const body = await request.json();

    const xaiResponse = await fetch('https://api.x.ai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + process.env.XAI_API_KEY,
      },
      body: JSON.stringify({
        ...body,
        stream: true, // Ensure streaming if client requests
      }),
    });

    if (!xaiResponse.ok) {
      const error = await xaiResponse.text();
      return new NextResponse(`Thunder error: ${xaiResponse.status} - ${error}`, {
        status: xaiResponse.status,
        headers: corsHeaders(),
      });
    }

    return new NextResponse(xaiResponse.body, {
      headers: {
        'Content-Type': 'application/json',
        ...corsHeaders(),
      },
    });

  } catch (err) {
    return new NextResponse('Lattice error: ' + err.message, {
      status: 500,
      headers: corsHeaders(),
    });
  }

  function corsHeaders() {
    return {
      'Access-Control-Allow-Origin': 'https://rathor.ai',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };
  }
}
