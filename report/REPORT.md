# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:**
    - Python: 3.10.14
    - PyTorch: 2.7.0
    - Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
    ('468', 'cab', '0.510915', 'ImageNet-1K')
- Record này mô tả toàn ảnh như thế nào?
    Record này mô tả resolution ảnh ở 640x428, nhận diện được "cab", có class_id:"468" với độ tin cậy là ~51%. Vì có độ tin cậy cao nhất -> xếp rank 1
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    class list là do con người định nghĩa thông qua Taxonomy trước khi huấn luyện mô hình. Không phải do mô hình tự định nghĩa
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    Cần giữ cả 3 trường vì:
      class_id giúp máy tính lập chỉ mục, lưu trữ và tính toán hiệu quả.
      class_name giúp con người đọc hiểu trực tiếp ngữ nghĩa mà không cần tra bảng mã.
      taxonomy_name đóng vai trò namespace xác định hệ quy chiếu ngữ cảnh và nguồn gốc, tránh xung đột nhãn.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ:
      1. Tiêu chí chọn chủ thể chính: Dựa trên diện tích pixel lớn nhất, nằm ở trung tâm/tiền cảnh hoặc độ rõ nét.
      2. Thứ tự ưu tiên: Khi các vật thể cạnh tranh ngang nhau (ví dụ: ưu tiên người > phương tiện > cảnh nền).
      3. Quy cách nhãn: Giữ đơn nhãn (single-label) hay chuyển sang đa nhãn (multi-label) / nhãn bối cảnh (scene label).
      4. Quy tắc escalation: Quy trình gắn cờ chuyển reviewer khi gặp ca mơ hồ, hoặc chuyển sang bài toán Object Detection nếu cần định vị từng vật thể riêng biệt."
- Vì sao model score không phải ground truth?
    Model score chỉ là ước lượng xác suất thống kê nội tại của thuật toán dựa trên các trọng số đã học; nó có thể rất cao nhưng vẫn đoán sai hoặc thấp nhưng vật thể vẫn có thật.
    Ngược lại, Ground Truth là chân lý thực tế đã được con người kiểm chứng và phê duyệt theo Guideline, đóng vai trò là thước đo chuẩn để đánh giá mô hình chứ không thể lấy từ chính đầu ra của mô hình.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): 
    - ('person', '0.912625', [385.33, 69.24, 498.92, 348.92], 113.58, 279.68)

- Diễn giải vị trí box bằng lời:
    - Hộp bao quanh đối tượng 'person' có định dạng xyxy = [385.33, 69.24, 498.92, 348.92]
    - Cạnh trái (x_min = 385.33 px): cách mép trái bức ảnh khoảng 385 pixel.
    - Cạnh trên (y_min = 69.24 px): cách mép trên bức ảnh khoảng 69 pixel (bắt đầu từ phần đỉnh đầu).
    - Cạnh phải (x_max = 498.92 px): cách mép trái ảnh khoảng 499 pixel.
    - Cạnh dưới (y_max = 348.92 px): cách mép trên ảnh khoảng 349 pixel (kéo dài xuống chân).

- So sánh số prediction ở hai threshold:
    - Ở threshold = 0.35: Model phát hiện 11 vật thể (2 person, 5 bowl, 2 oven, 2 cup).
    - Ở threshold = 0.60: Model chỉ phát hiện 6 vật thể (2 person, 2 bowl, 2 oven).
    -> Số lượng dự đoán giảm gần một nửa (giảm 5 vật thể); các đối tượng nhỏ/mờ hơn như cup và bowl có score < 0.60 đã bị lọc bỏ hoàn toàn.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    Về độ bao phủ:
      + Threshold thấp (0.35): Độ bao phủ cao hơn, bắt được nhiều vật thể nhỏ/bị che khuất (như cup, bowl), giảm nguy cơ bỏ sót vật thể thật (giảm False Negative).
      + Threshold cao (0.60): Độ bao phủ giảm rõ rệt, mô hình chỉ giữ các dự đoán rất tự tin, bỏ sót nhiều vật thể thực tế trong bếp.
    Khối lượng reviewer cần xem:
      + Threshold thấp (0.35): Khối lượng reviewer rất lớn (tăng False Positives), phải xem xét nhiều box nhiễu, tốn công gỡ lỗi.
      + Threshold cao (0.60): Khối lượng giảm mạnh (tăng False Negatives), reviewer chỉ tập trung vào các vật thể chắc chắn, nhưng rủi ro bỏ sót cao.

- Đề xuất một quy tắc box chặt:
    - Đặt ngưỡng score tối thiểu (ví dụ: 0.60) để loại bỏ nhanh các dự đoán nhiễu.
    - Cả 4 cạnh của bounding box (x_min, y_min, x_max, y_max) phải tiếp xúc khít với các điểm pixel ngoài cùng có thể nhìn thấy được của vật thể (trên, dưới, trái, phải), không để lọt khoảng trống nền lớn hơn 2-3 pixel.
    - Hộp phải bao trọn toàn bộ các bộ phận gắn liền của vật thể (ví dụ với 'person': tính cả tóc, quần áo, giày dép; không cắt cụt mép) nhưng không được bao gồm các vật thể cầm nắm rời rạc nếu chúng thuộc class khác (như dao, chảo, cốc).

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    - Guideline cần quy định:
      + Ngưỡng phần trăm nhìn thấy tối thiểu (Visibility Threshold): Ví dụ chỉ gán nhãn nếu nhìn thấy >= 20% vật thể; nếu bị che khuất > 80% thì bỏ qua để tránh gây nhiễu.
      + Phạm vi bao phủ: Box chỉ vẽ bao quanh "phần nhìn thấy được" (Visible box) hay vẽ ước lượng luôn cả "phần bị che khuất" (Amodal box).
    - Escalation cần quyết định:
      + Khi vật thể bị che khuất phân tách thành 2 mảnh rời rạc (ví dụ người đứng sau một cột bếp): Cần gán 1 box lớn gộp chung hay 2 box riêng lẻ?
      + Khi vật thể bị cắt mép ảnh chỉ còn một phần nhỏ không đủ nhận dạng chắc chắn: Đẩy lên Reviewer/Lead quyết định giữ hay loại bỏ.
    

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    - instance_id: "kitchen-001"
    - class_name: "person"
    - score: 0.902345
    - Số điểm polygon: 46 điểm
    - Một phần polygon_xy: [[428.12, 69.45], [435.5, 71.2], [450.18, 78.64], [465.3, 90.1], ..., [428.12, 69.45]]

- Polygon bổ sung chi tiết gì so với box?
    - Pixel-level boundary: Trong khi Bounding Box chỉ là hình chữ nhật thô (chứa cả vật thể lẫn diện tích pixel nền/nhiễu xung quanh), Polygon bám sát từng đường viền hình học thực tế của vật thể (đường cong cơ thể, vành bát, tay nắm cửa, quần áo).
    - Foreground vs. Background: Tách tuyệt đối điểm ảnh nào thuộc về vật thể và điểm ảnh nào là cảnh nền.
    - Cung cấp hình học thực tế: Cho phép tính toán chính xác diện tích bề mặt thực, chu vi, độ lồi lõm và góc xoay/tư thế của vật thể, đặc biệt với các đối tượng có hình dạng bất quy tắc hoặc nằm nghiêng.

- `instance_id` dùng để làm gì và không phải loại ID nào?
    - `instance_id` là mã định danh duy nhất cho từng vật thể riêng biệt trong một bức ảnh.
    - Trong khi `sample_id` định danh cho bức ảnh, `class_id` định danh cho loại đối tượng, thì `instance_id` (ví dụ: `kitchen-001`, `kitchen-002`) dùng để phân biệt các vật thể cùng loại trong cùng bức ảnh.
    - Nếu không có `instance_id`, hệ thống không thể phân biệt được đâu là "người thứ nhất", "người thứ hai", dẫn đến không thể theo dõi hay gán nhãn riêng lẻ từng đối tượng khi có nhiều vật thể cùng loại.

- Đề xuất một quy tắc biên mask:
    - Đường viền polygon phải bám sát mép ngoài cùng có thể quan sát được của vật thể, sai số lệch (offset/gap) giữa cạnh mask và biên vật thể không được vượt quá 1-2 pixel ở mức zoom 100%.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    - Vùng mờ (Motion blur / Out of focus):
      + Guideline: Quy định vẽ ranh giới cắt tại điểm chuyển tiếp 50% độ tương phản (contrast midpoint) giữa vật thể và nền.
      + Escalation: Nếu vật thể bị mờ quá 30% diện tích không thể phân biệt ranh giới bằng mắt thường, gắn cờ đẩy Reviewer quyết định có vẽ ước lượng hay đánh dấu 'unclear_boundary'.
    - Vùng tiếp xúc (Touching instances):
      + Guideline: Tuyệt đối không để chồng lấn (overlap) giữa các mask; với 2 vật thể cùng lớp chạm sát nhau (như 2 cái bát xếp chồng, 2 người đứng sát), ranh giới phân tách phải đi theo khe rãnh nhìn thấy rõ nhất.
      + Escalation: Trường hợp hòa trộn màu sắc không thể phân tách ranh giới tiếp xúc, chuyển Reviewer quyết định tách theo phỏng đoán nhân trắc học hay gộp tạm thời.
    - Vùng che khuất (Occlusion):
      + Guideline: Chỉ phân đoạn phần nhìn thấy được (Visible mask); không vẽ phỏng đoán phần bị che khuất (trừ khi dự án yêu cầu Amodal segmentation).
      + Escalation: Khi một vật thể bị che khuất ở giữa làm chia cắt thành 2 vùng nhìn thấy tách rời, Reviewer/Guideline phải quyết định gán nhãn dạng Multi-polygon (1 instance duy nhất có nhiều đa giác) hay tách làm 2 instance riêng biệt.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Cấp ảnh (Image-level): 1 nhãn duy nhất cho toàn ảnh (`image_id` → `class_id`, `class_name` thuộc taxonomy ImageNet-1K). | Ảnh `traffic` có nhiều chủ thể (ô tô `cab`, tàu điện `streetcar`, người, cột đèn). Model chỉ đoán `cab` với score ~0.51, gây mơ hồ về chủ thể đại diện cho toàn ảnh. | Đọc kỹ guideline để chọn đúng chủ thể chính (dựa trên diện tích lớn nhất, ở trung tâm hoặc tiêu điểm); gán đúng class; nếu mơ hồ thì gắn cờ escalation. | Kiểm tra nhãn gán có đúng chủ thể chính theo quy tắc guideline không; đảm bảo không nhầm lẫn giữa các lớp gần nghĩa (`cab` vs `passenger car`). |
| Phát hiện vật thể | Cấp vật thể (Object-level): Mỗi object có 1 record gồm `class_id` và hộp tọa độ pixel `bbox_xyxy = [x_min, y_min, x_max, y_max]` cùng kích thước `bbox_width`, `bbox_height`. | Ảnh `kitchen` có các vật thể nhỏ hoặc bị che khuất một phần (như `bowl`, `cup`) dễ bị mô hình bỏ sót khi tăng threshold lên 0.60, hoặc box bị vẽ quá rộng lọt cảnh nền. | Vẽ bounding box bao khít 4 điểm cực trị của từng vật thể nhìn thấy được; không bỏ sót vật thể nhỏ/che khuất thỏa mãn guideline; tách riêng từng box cho vật thể cùng lớp. | Kiểm tra độ chặt của box (box tightness: không thừa nền > 2-3px, không cắt cụt chi tiết); rà soát độ bao phủ (không bỏ sót vật thể - False Negative); kiểm tra gán đúng class. |
| Instance segmentation | Cấp cá thể (Instance-level): Mỗi cá thể có `instance_id` riêng, `class_id`, `bbox_xyxy` và danh sách điểm đa giác `polygon_xy` khép kín theo pixel. | Ảnh `kitchen` có vùng tiếp xúc dính liền giữa người và bàn bếp, hoặc ranh giới giữa các chiếc bát xếp chồng/che khuất nhau khiến biên đa giác bị lem nhem hoặc hòa lẫn. | Dùng công cụ polygon chấm các điểm nút bám sát viền thực của vật thể (sai số <= 1-2px); đục lỗ vùng rỗng xuyên thấu; gán đúng `instance_id` duy nhất cho từng cá thể. | Soi kỹ biên mask ở mức zoom lớn (đảm bảo không lấn ra nền, không lẹm vào trong, không chồng lấn giữa các mask); kiểm tra `instance_id` là duy nhất và đa giác khép kín hợp lệ. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    Không tải lên dữ liệu nội bộ, ảnh cá nhân, thông tin định danh (PII) hoặc dữ liệu nhạy cảm lên môi trường công khai (Colab public, GitHub public repo); chỉ sử dụng dữ liệu mẫu công khai được cấp phép.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    Giảng viên / Lab Coach / Người phụ trách quản trị dữ liệu của dự án
## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
