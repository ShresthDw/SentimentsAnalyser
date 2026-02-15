const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 5000;
const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json({ limit: '1mb' }));

app.get('/api/health', (_request, response) => {
  response.json({ status: 'ok' });
});

app.post('/api/predict', async (request, response) => {
  try {
    const targetResponse = await fetch(`${mlServiceUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request.body)
    });

    const payload = await targetResponse.json();
    response.status(targetResponse.status).json(payload);
  } catch (error) {
    response.status(500).json({
      error: 'Unable to reach the ML service.',
      details: error.message
    });
  }
});

app.listen(port, () => {
  console.log(`Backend listening on http://127.0.0.1:${port}`);
});