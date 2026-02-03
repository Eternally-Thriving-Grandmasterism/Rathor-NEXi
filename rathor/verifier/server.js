require('dotenv').config();
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 8081;
const HC_SECRET = process.env.HCAPTCHA_SECRET; // your hCaptcha secret key

app.post('/verify', async (req, res) => {
  const { token } = req.body;

  if (!token) {
    return res.status(400).json({ success: false, message: 'No token provided' });
  }

  try {
    const response = await axios.post('https://hcaptcha.com/siteverify', null, {
      params: {
        secret: HC_SECRET,
        response: token
      }
    });

    const data = response.data;

    if (data.success) {
      // Mercy gate: score threshold (hCaptcha returns score 0.0–1.0)
      const score = data.score || 0;
      if (score >= 0.7) {
        return res.json({ success: true, score });
      } else {
        return res.status(403).json({ success: false, message: 'Mercy shield: low CAPTCHA score', score });
      }
    } else {
      return res.status(403).json({ success: false, message: 'CAPTCHA failed', errors: data['error-codes'] });
    }
  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, message: 'Verifier error' });
  }
});

app.listen(PORT, () => {
  console.log(`Rathor hCaptcha Verifier running on port ${PORT}`);
});
