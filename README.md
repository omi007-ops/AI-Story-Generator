# 🧠 AI Creative Story Generator

A Generative AI pipeline built with **Google Gemini 2.5 Flash API** that generates creative stories, analyzes their emotional tone using NLP, and visualizes sentiment patterns.

## ✨ Features
- 📖 Generates genre-specific stories (Sci-Fi, Thriller, Romance, Fantasy, Mystery)
- 🎭 Full prompt engineering with style, tone & POV control
- 📊 Sentence-level sentiment analysis using TextBlob
- 📈 Visualizations: Emotional arc, sentiment donut chart, polarity scatter plot
- 🔁 Multi-genre batch comparison
- 💾 Exports story + analysis report to `.txt`

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| NLP | TextBlob |
| Visualization | Matplotlib |
| Language | Python 3.10+ |

## 🚀 How to Run
1. Open in [Google Colab](https://colab.research.google.com/)
2. Get a free API key at [aistudio.google.com](https://aistudio.google.com)
3. Add it to Colab Secrets as `GEMINI_API_KEY`
4. Run all cells top to bottom

## 📊 Output
- `story_analysis.png` — emotional arc & sentiment charts
- `genre_comparison.png` — cross-genre sentiment comparison
- `<story_title>.txt` — exported story with analysis report
