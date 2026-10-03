# Báo cáo Day 17: Memory Systems for AI Agent

## Trạng thái

- Mã nguồn và test đã xong: `pytest src/test_agents.py -v` (32 test, chạy không cần API key).
- **Đã chạy benchmark thật** bằng `python src/benchmark.py` với OpenRouter, model `openai/gpt-4o-mini` (cả agent lẫn judge), nhiệt độ 0, `LLM_MAX_TOKENS=1024`. Số liệu ở mục 3 là lần chạy **sau** khi sửa parser. Mỗi suite chỉ chạy **một lần**; lần chạy trước khi sửa cho số khác (xem mục 4), nên chênh lệch giữa các lần là có thật và chưa đo được độ dao động.
- Một bản làm trước đó dùng regex và danh sách từ khoá theo dữ liệu mẫu (recall 1.00). Mình đã **gỡ bỏ** nó vì đó là hard code: nó chứng minh cơ chế chạy, không chứng minh khả năng tổng quát. Các con số của bản đó không còn dùng.

## 1. Thiết kế ba lớp memory

| Lớp | Nằm ở đâu | Sống bao lâu | Chứa gì |
|---|---|---|---|
| Short-term | `CompactMemoryManager.messages` | trong một thread | các message gần nhất, nguyên văn |
| Persistent | `state/profiles/<user>/User.md` | qua mọi thread và lần chạy | fact ổn định về người dùng, schema tự do (key do LLM đặt) |
| Compact | `CompactMemoryManager.summary` | trong một thread | bản tóm tắt cuộn do LLM viết, tối đa 10 dòng |

Một lượt của Advanced:
1. LLM nhận `User.md` hiện tại + tin nhắn mới và trả JSON `{updates, remove}` kèm `confidence` cho mỗi fact.
2. Code chỉ ghi fact có `confidence >= PROFILE_MIN_CONFIDENCE` (mặc định 0.7). Fact mới cùng key sẽ **ghi đè** fact cũ.
3. Tin nhắn vào short-term; nếu vượt `COMPACT_THRESHOLD_TOKENS` thì phần cũ được LLM tóm tắt.
4. Prompt = `User.md` + summary + message gần nhất, rồi LLM trả lời.

Baseline chỉ gửi toàn bộ thread hiện tại, không có `User.md`, nên thread mới bắt đầu trống.

Không còn quy tắc nào gắn với dữ liệu: không regex, không danh sách từ khoá, không mẫu câu trả lời. Việc phân biệt fact thật với câu hỏi, câu đùa, giả định do prompt trích fact (`EXTRACTION_PROMPT`) giao cho LLM. Hai tham số điều chỉnh được qua biến môi trường: `PROFILE_MIN_CONFIDENCE`, `COMPACT_THRESHOLD_TOKENS` / `COMPACT_KEEP_MESSAGES`.

## 2. Test kiểm chứng gì, và không kiểm chứng gì

Test dùng LLM giả có kịch bản nên **chỉ kiểm tra đường dẫn dữ liệu**: fact đã lưu có đến prompt của thread mới không; baseline có bị rò sang thread khác không; correction có ghi đè không; fact dưới ngưỡng tin cậy có bị bỏ không; JSON hỏng có không ghi gì không; summary có vào prompt sau compact không; chi phí trích fact/tóm tắt có được tính riêng không. Mình đã phá thử mã nguồn để chắc các test này đỏ khi có lỗi (và tìm ra, rồi vá, một lỗ hổng: chưa có test cho việc summary đến được prompt).

Test **không** kiểm chứng chất lượng trích fact của một model thật. Phần đó chỉ đo được qua benchmark.

## 3. Kết quả benchmark (`python src/benchmark.py`, một lần chạy, sau khi sửa parser)

**Standard Benchmark**

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 5260 | 31998 | 0.11 | 0.14 | 0 | 0 |
| Advanced | 4960 | 40373 | 0.93 | 0.93 | 610 | 2 |

Prompt tokens Advanced so với Baseline: +26.2%. Ngoài bảng, Advanced tiêu thêm ~50492 token cho trích fact và tóm tắt.

**Long-Context Stress Benchmark**

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 3331 | 43708 | 0.00 | 0.27 | 0 | 0 |
| Advanced | 2095 | 13208 | 1.00 | 1.00 | 277 | 25 |

Prompt tokens Advanced so với Baseline: -69.8%. Ngoài bảng, Advanced tiêu thêm ~28909 token cho trích fact và tóm tắt.

Lần chạy trước khi sửa parser (cũng một lần): Standard recall 0.96, prompt +32.8%; Stress recall 0.67, prompt -68.6%, 27 compaction.

## 4. Đọc kết quả

- **Recall:** đúng hướng giả thuyết. Baseline gần như không nhớ gì qua thread mới (0.11 và 0.00; phần nhỏ còn lại là đáp án tình cờ nằm trong chính câu hỏi). Advanced đạt 0.93 ở Standard và 1.00 ở Stress, kể cả correction (Huế → Đà Nẵng) và câu đùa gây nhiễu (product manager). `User.md` cuối của Stress đúng: tên, `Đà Nẵng`, `MLOps engineer`, style.
- **Hội thoại ngắn (Standard):** Advanced tốn prompt nhiều hơn 26.2%: mỗi lượt mang thêm `User.md`, còn compact chỉ chạy 2 lần. Cộng cả ~50k token gọi phụ thì chi phí tăng nhiều hơn con số đó. Đây là cái giá của memory khi hội thoại ngắn.
- **Hội thoại dài (Stress):** prompt giảm 69.8% (tiết kiệm 43708 - 13208 = 30500), nhưng các lần gọi trích fact và tóm tắt tốn ~28909 token, gần bằng mức tiết kiệm. Tính theo tổng token thì gần hoà vốn; lợi ích còn lại nằm ở độ trễ và giá token vào/ra khác nhau, và ở chỗ Baseline không nhớ gì còn Advanced nhớ đủ. Compact chỉ giảm `Prompt tokens processed`, không giảm `Agent tokens only`.
- **Stress recall đã tăng từ 0.67 lên 1.00, nhưng chưa thể quy cho bản sửa parser.** Giữa hai lần chạy mình sửa parser (nhận giá trị trần, nối danh sách), nhưng model cũng không tất định: chạy lại riêng hội thoại này 3 lần cho 3 kết quả khác nhau (một lần model đổi tên key `location` thành `residence`). Lần chạy gốc mất cả nghề nghiệp lẫn nơi ở không tái hiện được và nguyên nhân chưa rõ. Cần nhiều lần chạy mới kết luận được.
- **Lỗi thật đã sửa:** có lượt model trả giá trị trần không bọc `{value, confidence}`; parser cũ bỏ im lặng, và danh sách bị lưu dạng `"['a', 'b']"`. Giờ nhận giá trị trần, nối danh sách thành `a, b`; có test.
- **25 compaction trên 16 lượt là quá nhiều.** Tin nhắn của dataset này dài (~145 token), nên ngưỡng 800 bị vượt gần như mỗi lượt và mỗi lần compact tốn một lần gọi LLM. Ngưỡng mặc định 800 / giữ 4 message chưa được tối ưu; chưa quét lại với model thật.
- **Chất lượng `User.md` ở Standard khá hơn nhưng còn thô:** giá trị đã là danh sách gọn, nhưng vẫn có nhiều key cho cùng một ý (`habits` gộp cả sở thích uống cà phê lẫn `favorite_drink`; `interests` lẫn mục không phải sở thích như "ví dụ gắn với công việc MLOps") và `pets: corgi named Bơ` lẫn tiếng Anh. Recall cao vì câu trả lời vẫn chứa đủ từ khoá.
- **Response quality** do judge LLM chấm, cùng model với agent nên có thể thiên vị; nó bám sát recall (0.14 → 0.93), không cung cấp nhiều thông tin độc lập.

## 5. Rủi ro của thiết kế

- **Lưu sai fact là lỗi âm thầm và bền:** một lần LLM hiểu nhầm sẽ đi theo người dùng qua mọi thread sau. Confidence threshold giảm rủi ro này nhưng độ tin cậy do chính LLM tự báo, nên không phải bảo đảm.
- **Đánh đổi của ngưỡng:** ngưỡng cao hơn ít ghi sai nhưng bỏ sót fact thật (recall giảm); ngưỡng thấp thì ngược lại.
- **Conflict handling theo kiểu "lần nói sau thắng":** không có bước xác nhận với người dùng. Fact dạng danh sách (sở thích, style) do LLM tự gộp giá trị cũ và mới; nếu nó gộp sai thì mất thông tin.
- **Summary có mất mát** và cũng do LLM viết: chi tiết cụ thể có thể rơi. Chấp nhận được vì fact ổn định nằm ở `User.md`.
- **`User.md` là dữ liệu cá nhân dạng rõ:** cần quyền truy cập và chính sách xoá nếu dùng thật. Kích thước file bị chặn theo số *key* chứ không theo số lượt (key trùng thì ghi đè), nhưng nếu LLM đặt key không nhất quán (`city`, `location`, `noi_o`) thì cùng một fact có thể nằm ở nhiều dòng. Prompt trích fact đưa profile hiện tại vào để model dùng lại key cũ, nhưng không có gì ép buộc.
- **Chi phí và độ trễ:** mỗi lượt Advanced có thêm một lần gọi LLM.

## 6. Giới hạn

- Mỗi bảng là **một lần chạy** với một model nhỏ (`gpt-4o-mini`) qua OpenRouter. Hai lần chạy (trước và sau sửa parser) lệch nhau đáng kể (Stress recall 0.67 vs 1.00), nên không nên coi chênh lệch nhỏ là có ý nghĩa; chưa lặp lại để có khoảng dao động.
- Dataset nhỏ (10 + 1 hội thoại cố định), nên 0.93 hay 0.96 chỉ là vài câu hỏi.
- Token là ước lượng (`ceil(len/4)`), không phải tokenizer thật; tỉ lệ giữa các agent đáng tin hơn con số tuyệt đối.
- `Response quality` do judge LLM chấm (cùng model với agent); heuristic (80% số ý đúng + 20% độ ngắn) chỉ là dự phòng khi judge trả về giá trị không hợp lệ.
- Chưa quét lại ngưỡng compact với model thật; chưa xác định được vì sao lần chạy gốc của Stress mất nghề nghiệp/nơi ở.
- Chưa làm memory decay.
