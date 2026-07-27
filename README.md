# Hệ thống gợi ý nhà hàng (Zomato Bangalore)

Hệ thống gợi ý nhà hàng lai (hybrid) dựa trên dữ liệu [Zomato Bangalore Restaurants](https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants). Người dùng chọn khu vực, món ăn, ngân sách và khoảng cách tối đa; hệ thống lọc theo ràng buộc, tính độ tương đồng nội dung (TF-IDF) và xếp hạng có trọng số. Giao diện Streamlit hỗ trợ geocode địa chỉ (OpenStreetMap Nominatim) và ước lượng quãng đường/thời gian lái xe (OSRM).

## Tính năng chính

- Lọc theo ngân sách, khoảng cách và loại ẩm thực
- Khoảng cách Haversine theo khu vực Bangalore
- Độ tương đồng nội dung (TF-IDF) trên `cuisines`, `rest_type`, `location`, `reviews_list`
- Điểm xếp hạng Bayesian (giảm thiên lệch khi ít lượt vote)
- Xếp hạng lai với trọng số mặc định:

```text
final_score =
    0.35 * rating_score
  + 0.25 * distance_score
  + 0.20 * price_score
  + 0.10 * popularity_score
  + 0.10 * content_similarity_score
```

- Giao diện web Streamlit: tìm nhà hàng, xem bản đồ và khoảng cách đường bộ

## Cấu trúc dự án

| File / thư mục                                     | Mô tả                                           |
| -------------------------------------------------- | ----------------------------------------------- |
| `streamlit_app.py`                                 | Giao diện web chính                             |
| `hybrid_recommender.py`                            | Mô hình gợi ý lai                               |
| `map_services.py`                                  | Geocode (Nominatim) và khoảng cách đường (OSRM) |
| `requirements.txt`                                 | Thư viện Python cần thiết                       |
| `data/zomato.csv`                                  | Dataset (tự tải, không có sẵn trong repo)       |
| `Zomato.ipynb`, `ZomatoRecommendationSystem.ipynb` | Notebook phân tích / baseline (tùy chọn)        |

### File được dùng khi chạy hệ thống

```text
streamlit_app.py
  ├── hybrid_recommender.py
  ├── map_services.py
  └── data/zomato.csv

python hybrid_recommender.py
  └── data/zomato.csv
```

## Thư viện (`requirements.txt`)

**Bắt buộc** (để chạy app Streamlit / CLI):

| Thư viện       | Dùng cho                    |
| -------------- | --------------------------- |
| `pandas`       | Đọc và xử lý dataset        |
| `numpy`        | Tính khoảng cách, điểm số   |
| `scikit-learn` | TF-IDF và cosine similarity |
| `streamlit`    | Giao diện web               |
| `requests`     | Gọi Nominatim / OSRM        |
| `pydeck`       | Bản đồ trong Streamlit      |

**Tùy chọn** (chỉ cần nếu mở notebook):

| Thư viện     | Dùng cho                          |
| ------------ | --------------------------------- |
| `matplotlib` | Vẽ biểu đồ EDA                    |
| `seaborn`    | Vẽ biểu đồ EDA                    |
| `nltk`       | Xử lý stopwords trong notebook cũ |

## Yêu cầu

- Python 3.10+ (khuyến nghị)
- Kết nối internet (geocode và OSRM khi dùng giao diện Streamlit)
- Dataset Kaggle `zomato.csv`

## Cách clone và chạy

### 1. Clone repository

Nếu bạn có URL GitHub của repo:

```bash
git clone <URL-repo>
cd Restaurant-Recommendation-System-main
```

Nếu đã tải ZIP, giải nén rồi mở thư mục dự án trong terminal:

```bash
cd Restaurant-Recommendation-System-main
```

### 2. Tạo môi trường ảo và cài thư viện

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Tải và đặt dataset

1. Tải dataset từ Kaggle: [Zomato Bangalore Restaurants](https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants/download)
2. Tạo thư mục `data` trong gốc dự án
3. Đặt file CSV vào đúng đường dẫn:

```text
data/zomato.csv
```

Ví dụ trên Windows:

```powershell
mkdir data
# Sao chép file zomato.csv đã tải vào thư mục data\
```

### 4. Chạy giao diện Streamlit (khuyến nghị)

```bash
streamlit run streamlit_app.py
```

Mở trình duyệt tại địa chỉ Streamlit hiển thị (thường là `http://localhost:8501`).

**Cách dùng nhanh:**

1. Chọn **Area** (khu vực hiện tại)
2. Nhập **Your address** (địa chỉ chi tiết hơn, tùy chọn)
3. Chọn một hoặc nhiều **Cuisines**
4. Đặt ngân sách (INR) — bật **Budget per person** nếu ngân sách là theo người
5. Đặt khoảng cách lái xe tối đa (km)
6. Bấm **Find restaurants**

Trang chủ cũng hiện các nhà hàng phổ biến trong khu vực đã chọn.

### 5. Chạy mô hình từ dòng lệnh (không UI)

```bash
python hybrid_recommender.py
```

Script dùng cấu hình mẫu: ngân sách 500 INR (cho 2 người), bán kính 3 km, món Biryani, khu vực Koramangala, top 10 kết quả.

### 6. Dùng trong Python / notebook

```python
from hybrid_recommender import (
    UserPreference,
    evaluate_rule_checking,
    load_zomato_data,
    recommend_restaurants,
)

restaurants = load_zomato_data("data/zomato.csv")

preference = UserPreference(
    budget=500,
    max_distance_km=3,
    cuisine="Biryani",
    current_area="Koramangala",
    top_n=10,
)

recommendations = recommend_restaurants(restaurants, preference)
evaluation = evaluate_rule_checking(recommendations, preference)

recommendations
```

Trường chi phí trong dataset là `approx_cost(for two people)`. Nếu ngân sách của bạn là **theo người**, đặt `budget_is_per_person=True`.

## Lưu ý

- Thư mục `data/` nằm trong `.gitignore` — mỗi máy cần tự tải dataset.
- Geocode/OSRM dùng API công cộng; lần tìm đầu có thể chậm hơn vì cần geocode nhiều địa chỉ (kết quả được cache bởi Streamlit).
- Notebook chỉ để tham khảo, không bắt buộc để chạy ứng dụng Streamlit.
