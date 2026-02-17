import "dotenv/config";

const baseUrl = process.env.REPLAY_LAB_API_URL;
const apiKey = process.env.REPLAY_LAB_API_KEY;

if (!baseUrl || !apiKey) {
  console.error("Missing REPLAY_LAB_API_URL or REPLAY_LAB_API_KEY in .env");
  process.exit(1);
}

const healthUrl = `${baseUrl.replace(/\/$/, "")}/api/health`;

const response = await fetch(healthUrl, {
  headers: {
    "x-api-key": apiKey,
  },
});

if (!response.ok) {
  const body = await response.text();
  console.error(`Replay Lab API health check failed: ${response.status}`);
  console.error(body);
  process.exit(1);
}

const data = await response.json();
console.log("Replay Lab API health:", data);
