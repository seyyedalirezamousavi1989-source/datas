# آزمایشگاه لبه (Edge Lab) و مدل AURORA در LocalChat 2.0.19

## چه چیزی اضافه شد
- `local_chat/edge_lab.py`: جست‌وجوی صادقانهٔ لبه (غربال جفت‌ارز، ۲۱۶ پیکربندی ثابت، walk-forward با purge، آزمون SPA، کنترل رانش (timing null)، تست هزینهٔ دوبرابر، تکرارپذیری بین نمادها، پنجرهٔ مهرشده که فقط یک‌بار باز می‌شود).
- `aurora_trade_patterns.py` / `aurora_trades_v3.py`: برچسب‌زدن معاملات و استخراج الگو (BH-FDR + تأیید روی بازهٔ بعد) و دیتاست v3.
- ابزار ایجنت `get_edge_report`: آخرین گزارش را از `%LOCALAPPDATA%\LocalChat\edge_lab\latest_digest.txt` می‌خواند؛ اگر نباشد صادقانه می‌گوید گزارشی نیست.
- حالت AURORA فشرده (پیش‌فرض خاموش): `LOCALCHAT_COMPACT_AGENT=1` فقط با مدل fine-tune‌شده؛ `LOCALCHAT_FOREX_SMC=1` بلوک SMC را به تحلیل اضافه می‌کند.

## اجرا (روی کامپیوتر خودت با MT5 باز)
```
python -m local_chat.edge_lab screen --mt5
python -m local_chat.edge_lab search --mt5 --out "%LOCALAPPDATA%\LocalChat\edge_lab"
python -m local_chat.edge_lab holdout --mt5 --out "%LOCALAPPDATA%\LocalChat\edge_lab" --finalist <id>   # فقط یک بار
```
حکم‌ها: `no_edge` (پیش‌فرض)، `candidate` (اثبات‌نشده)، `holdout_passed` (باید به تست رو‌به‌جلو/دمو برود، نه سرمایه).

## آموزش مدل
`training_data/aurora_unsloth/aurora_compact_v3_*.jsonl` + `train_unsloth_qwen.py` (راهنما: FINETUNING_GUIDE_FA.md). هیچ مدلی آموزش داده نشده است و لبهٔ معتبری ثابت نشده؛ نتیجهٔ XAUUSD: no_edge با یک نامزد نزدیک (خرید ساعت ۲۰–۲۲ سرور، SPA p=0.06).
