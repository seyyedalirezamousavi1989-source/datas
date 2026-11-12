# AURORA QUANT — سیستم هویت، آگاهی و حافظهٔ پایدار

وضعیت: پیاده‌سازی و تست‌شده با mock/واحد؛ **هیچ مدلی آموزش داده نشده و وزنی تغییر نکرده است.** چیزی به GitHub پوش نشده است.

## ۱. بازرسی معماری موجود و بازاستفاده (بدون ساخت سیستم تکراری)
| نیاز | موجود در LocalChat | کار انجام‌شده |
|---|---|---|
| احراز سازنده | `server.py`: کد ۶‌رقمی + PBKDF2 + تلاش/انقضا + SQLite پایدار | **دست‌نخورده**؛ فقط نتیجه‌اش (`_registered_creator()`) به نقش تبدیل می‌شود |
| رجیستری قابلیت | `capabilities.py` (id، شرح، ورودی/خروجی، وابستگی، محدودیت، `availability()`) | دو تابع read-only اضافه شد (`iter_capabilities`, `get_capability`)؛ فیلدهای کم (وضعیت ۷‌حالته، نسخه، آخرین راستی‌آزمایی) در یک overlay JSON |
| حافظه | `memory.py` (SQLite، متن‌های ذخیره‌شدهٔ کاربر) | همان فایل دیتابیس؛ جدول نوع‌دار `identity_facts` (نوع/وضعیت/جایگزینی/شاهد) کنار `memories` |
| پرامپت هویت | `_local_environment_facts` / `_system_prompt` | بلوک کوتاه `[هویت AURORA]` از وضعیت واقعی برنامه ساخته و اضافه می‌شود؛ `_IDENTITY_CUE` گسترش یافت |

## ۲. هویت کانونی
سازنده/بنیان‌گذار/توسعه‌دهندهٔ اصلی: **Alireza Mousavi** (علیرضا موسوی) — ایجنت: **AURORA QUANT** — سازمان: **AURORA AI Research Group** — پایهٔ فنی: **LocalChat**. مدل زبانی = موتور استدلال و جدا از ایجنت. ساختار سازمان، کارکنان، انتشارات، دستاوردها و تاریخچه **نامعلوم** ثبت شده‌اند (نه صفر).

## ۳. ماژول‌ها (`local_chat/`)
- `aurora_identity.py`: `CANON`، `role_from_app_state`، `authorize` (چهار لایه: شناسایی، احراز، مجوز، ایمنی)، `CapabilityOverlay` + `capability_status` + `relevant_capabilities`، `FactStore`، `identity_context`.
- `aurora_dataset.py`: ساخت ۹ فایل JSONL + `export_sft` (فرمت `messages` برای Unsloth/TRL).
- `aurora_eval.py`: ممیزی دیتاست + `evaluate(generate, dir, split)` برای هر مدل (قانون‌محور).
- `test_aurora_identity.py`: ۳۲ تست.

### مرزهای اختیار (`authorize`)
ادعای «من سازنده‌ام» نادیده گرفته می‌شود. سازندهٔ تأییدشده: پیکربندی/بازرسی/توسعه مجاز؛ عملیات مخرب نیازمند تأیید؛ غیرقابل‌بازگشت نیازمند مجوز صریح؛ **دور زدن کنترل امنیتی و افشای راز برای هیچ نقشی مجاز نیست**؛ شکست اعتبارسنجی ایمنی حتی سازنده را متوقف می‌کند.

### وضعیت قابلیت‌ها
هفت وضعیت: available, active, disabled, unavailable, requires_configuration, requires_authorization, experimental. Override فقط می‌تواند محدودتر کند؛ قابلیتی که `availability()` آن False است هرگز با override «در دسترس» نمی‌شود. فقط قابلیت‌های مرتبط با همان درخواست (حداکثر ۴) وارد پرامپت می‌شوند.

### حافظهٔ نوع‌دار
انواع: confirmed_fact, user_provided_unverified, hypothesis, recommendation, future_plan, decision, implemented, tested, active_task, completed_work, pending_work, known_limitation. پیشنهاد/فرضیه/برنامه **مستقیماً** به implemented/tested/confirmed تبدیل نمی‌شود؛ `implemented/tested/completed_work` بدون شاهد پذیرفته نمی‌شود؛ تصمیم جدید قدیمی را `superseded` می‌کند و تاریخچه حفظ می‌شود؛ رازها، ایمیل و کدِ شش‌رقمی ذخیره نمی‌شوند.

## ۴. دیتاست (`training_data/aurora_identity/`)
۱٬۱۵۶ نمونه (۵۷۸ فارسی + ۵۷۸ انگلیسی)؛ train 772 / val 192 / test 192. تقسیم: در هر intent دو عبارت‌بندی آخر برای val و test نگه داشته می‌شود (عبارت مشترک بین splitها نیست).
| فایل | نمونه |
|---|---|
| creator_identity | 50 |
| agent_identity | 48 |
| organization_identity | 50 |
| creator_relationship | 96 |
| identity_security | 48 |
| capability_awareness | 600 |
| model_vs_agent_identity | 60 |
| persistent_memory_awareness | 108 |
| identity_integrated | 96 |

سیستم‌پیام هر نمونه با **همان تابع runtime** (`identity_context`) ساخته شده است. ممیزی (۱۵ بررسی): بدون ایمیل/کد/رمز/توکن، شناسه‌های یکتا، برچسب نقش = وضعیت سیستم‌پیام، هرگز نقش سازنده برای کاربر تأییدنشده، فقط نام‌های کانونی، وضعیت قابلیت = زمینه، قابلیت نقشهٔ راه ادعا نشده، بدون اشتراک عبارت بین splitها، پاسخ مرجع با قوانین خودش سازگار.

## ۵. محدودیت‌ها و موارد باز (صادقانه)
1. **دیتاست شناخت هویت و رفتار را آموزش می‌دهد، نه تحلیل دقیق فارکس.** مهارت تحلیل از دیتاست‌های XAUUSD مرحلهٔ ۵ می‌آید و آن ارزیابی‌ها مهارت پیش‌بینی تأییدشده نشان ندادند (AUC حدود ۰٫۵). دقت یا سودآوری وعده داده نمی‌شود.
2. حجم دیتاست هویت کوچک و قالبی (template) است؛ ۵۲٪ آن قابلیت‌هاست. تنوع زبانی محدود است؛ برای آموزش واقعی بهتر است با دیتاست مرحلهٔ ۵ ترکیب شود و نسبت هویت کم بماند.
3. مدل آموزش‌دیده‌ای وجود ندارد، پس **ارزیابی مدل واقعی انجام نشده**؛ فقط ابزار ارزیابی و ممیزی دیتاست تست شده است. معیارها قانون‌محور و شرط لازم‌اند نه کافی.
4. احراز سازنده در LocalChat **سطح دستگاه** است (بدون نشست/نقش چندکاربره): پس از تأیید، هر کس پشت همان نصب نقش سازنده دارد. `authorize()` هنوز به endpointهای مخرب موجود (system_tools، حذف حافظه و …) **وصل نشده**؛ آن endpointها تأیید خودشان را دارند ولی باید یکی‌شان کرد.
5. **ایمیل واقعی سازنده در `server.py` (`CREATOR`) به‌صورت ثابت در سورس است.** در پرامپت و دیتاست نمی‌آید، اما اگر این مخزن عمومی شود افشا می‌شود؛ پیشنهاد: انتقال به فایل تنظیم محلی/gitignore.
6. نام فارسی «سید علیرضا موسوی» در `CREATOR` با نام کانونی «Alireza Mousavi» به‌عنوان نام مستعار هم‌خوان فرض شده؛ لطفاً تأیید کنید.
7. اتصال MT5/LocalChat واقعی روی ویندوز اجرا نشد؛ تست‌ها واحد و mock هستند. نام مدل‌های نمونه (`qwen3.5:4b` …) فقط مقدار نمونهٔ runtime است و تأیید نشده که در Ollama شما وجود دارند.
8. آموزش: فقط دستور؛ هیچ آموزشی اجرا نشد (مثلاً با `FastLanguageModel` + `train_on_responses_only` روی `training_data/aurora_identity/sft/identity_sft_train.jsonl`، ارزیابی روی val، و `evaluate()` با خروجی مدل).

## ۶. نتایج تست
`python -m pytest local_chat` → **623 passed, 1 skipped** (۳۲ تست جدید AURORA؛ بقیه از قبل). یک باگ ترتیب‌دهی `FactStore.history` و یک حفرهٔ فیلتر راز (جملهٔ «verification code is …») در حین تست پیدا و رفع شد.
