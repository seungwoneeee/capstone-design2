from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, Date, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#from model import functionname

# 한글 폰트 경로 설정 (윈도우 기준, Mac이나 Linux 환경에서는 다를 수 있음)
# font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_path = "C:/Windows/Fonts/malgun.ttf"


# 한글 폰트 등록
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 데이터베이스 연결 설정
username = 'root'
password = 'tmddnjs7867'
host = 'localhost'
port = '3306'
database = 'capstone2'
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Naver_review와 Coupang_review 모델 정의
class NaverReview(Base):
    __tablename__ = "Naver_review"
    index = Column(Integer, primary_key=True, index=True)
    product_code = Column(String)
    product_name = Column(String)
    use_type = Column(String)
    star = Column(Integer)
    review = Column(String)
    review_recommend = Column(Integer, nullable=True)
    id = Column(String)
    review_date = Column(Date)
    best_review = Column(String)
    Predict_segment = Column(Integer, nullable=True)

class CoupangReview(Base):
    __tablename__ = "Coupang_review"
    index = Column(Integer, primary_key=True, index=True)
    review_date = Column(Date)
    product_code = Column(String)
    star = Column(Integer)
    review = Column(String)
    deliver_type = Column(String)
    Predict_segment = Column(Integer, nullable=True)

# FastAPI 애플리케이션 생성
app = FastAPI()
Base.metadata.create_all(bind=engine)

@app.post("/data/new")
async def add_data(
    review_type: str = Query(..., description="Type of review: coupang or naver"),
    file: UploadFile = File(...)
):
    if file.content_type != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()
    df = pd.read_excel(BytesIO(contents), sheet_name=None)  # 모든 시트를 읽기

    session = SessionLocal()
    try:
        if review_type == "coupang":
            all_reviews = []
            for sheet_name, sheet_df in df.items():
                sheet_df = sheet_df.copy()
                if sheet_name == '쿠팡_3P':
                    required_columns = ['등록일', '품목코드', '별점', '상품평 코멘트']
                    sheet_df = sheet_df[required_columns]
                    sheet_df.rename(columns={
                        '등록일': 'review_date',
                        '품목코드': 'product_code',
                        '별점': 'star',
                        '상품평 코멘트': 'review'
                    }, inplace=True)
                    sheet_df['deliver_type'] = '쿠팡_3P'
                elif sheet_name == '쿠팡_로켓':
                    required_columns = ['리뷰 일자', '품목코드', '상품별점', '리뷰내용']
                    sheet_df = sheet_df[required_columns]
                    sheet_df.rename(columns={
                        '리뷰 일자': 'review_date',
                        '품목코드': 'product_code',
                        '상품별점': 'star',
                        '리뷰내용': 'review',
                    }, inplace=True)
                    sheet_df['deliver_type'] = '쿠팡_로켓'
                else:
                    continue

                # NaN 값을 None으로 대체
                sheet_df = sheet_df.replace({np.nan: None})
                all_reviews.append(sheet_df)

            # 가장 큰 index 값을 가져오기
            last_index = session.query(func.max(CoupangReview.index)).scalar() or 0
            reviews = []

            # DataFrame의 각 행을 반복하며 리뷰 객체 생성 및 리스트에 추가
            for sheet_df in all_reviews:
                for i, row in enumerate(sheet_df.itertuples(), start=1):
                    review = CoupangReview(
                        index=last_index + i,  # index 값 증가
                        review_date=row.review_date,
                        product_code=row.product_code,
                        star=row.star,
                        review=row.review,
                        deliver_type=row.deliver_type
                    )
                    reviews.append(review)
                last_index += len(sheet_df)
            session.bulk_save_objects(reviews)
        elif review_type == "naver":
            required_columns = ['INDEX', '상품번호', '상품명', '리뷰구분', '구매자평점', '리뷰상세내용', '리뷰도움수', '등록자', '리뷰등록일', '베스트리뷰']
            sheet_df = df[list(df.keys())[0]]  # 첫 번째 시트가 Naver 관련 시트라고 가정
            for col in required_columns:
                if col not in sheet_df.columns:
                    sheet_df[col] = None
            sheet_df = sheet_df[required_columns]
            sheet_df.rename(columns={
                'INDEX': 'product_code',
                '상품명': 'product_name',
                '리뷰구분': 'use_type',
                '구매자평점': 'star',
                '리뷰상세내용': 'review',
                '리뷰도움수': 'review_recommend',
                '등록자': 'id',
                '리뷰등록일': 'review_date',
                '베스트리뷰': 'best_review'
            }, inplace=True)

            # NaN 값을 None으로 대체
            sheet_df = sheet_df.replace({np.nan: None})
            # 데이터베이스에서 마지막 index 값을 가져오기
            last_index = session.query(func.max(NaverReview.index)).scalar() or 0
            reviews = []  # naver 리뷰를 위한 reviews 리스트 초기화

            # 자동 증가된 index 값을 사용하여 일괄 삽입
            for i, row in enumerate(sheet_df.itertuples(), start=1):
                review = NaverReview(
                    index=last_index + i,  # index 값 증가
                    product_code=row.product_code,
                    product_name=row.product_name,
                    use_type=row.use_type,
                    star=row.star,
                    review=row.review,
                    review_recommend=row.review_recommend,
                    id=row.id,
                    review_date=row.review_date,
                    best_review=row.best_review
                )
                reviews.append(review)
            session.bulk_save_objects(reviews)
        else:
            raise HTTPException(status_code=400, detail="Invalid review type")

        session.commit()
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to insert data: {str(e)}")
    finally:
        session.close()

    return JSONResponse(status_code=200, content={"message": "Data added successfully"})

# AI 모델 불러오는 함수
def update_predict_segment(review_type: str):
    session = SessionLocal()
    if review_type == "coupang":
        reviews = session.query(CoupangReview).filter(CoupangReview.Predict_segment == None).all()
    elif review_type == "naver":
        reviews = session.query(NaverReview).filter(NaverReview.Predict_segment == None).all()
    else:
        session.close()
        return

    # 임시 AI 모델
    for review in reviews:
        # 더미 예측 로직
        review.Predict_segment = 1#functionname(review)  # 실제 예측으로 교체

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update Predict_segment: {str(e)}")
    finally:
        session.close()

# 데이터베이스에 실제로 리뷰 데이터가 존재하는지 확인하는 API
@app.get("/check_reviews_existence")
def check_reviews_existence():
    db = SessionLocal()
    naver_reviews_exist = db.query(NaverReview).count() > 0
    coupang_reviews_exist = db.query(CoupangReview).count() > 0
    db.close()
    return {"naver_reviews_exist": naver_reviews_exist, "coupang_reviews_exist": coupang_reviews_exist}

@app.get("/naver_reviews")
def get_naver_reviews():
    db = SessionLocal()
    # Naver 리뷰 데이터 가져오기
    naver_reviews = db.query(NaverReview.star, NaverReview.review, NaverReview.id).all()
    naver_result = [{"star": star, "reviews": [{"review": review, "id": user_id} for _, review, user_id in naver_reviews if _ == star]} for star in range(1, 6)]
    # Naver 별점 분포 데이터 가져오기
    naver_star_counts = db.query(NaverReview.star, func.count(NaverReview.star)).group_by(NaverReview.star).all()
    naver_star_result = [{"star": star, "count": count} for star, count in naver_star_counts]
    db.close()
    return {"naver_reviews": naver_result, "naver_star_counts": naver_star_result}

@app.get("/coupang_reviews")
def get_coupang_reviews():
    db = SessionLocal()
    # Coupang 리뷰 데이터 가져오기
    coupang_reviews = db.query(CoupangReview.star, CoupangReview.review, CoupangReview.deliver_type).all()
    coupang_result = [{"star": star, "reviews": [{"review": review, "deliver_type": deliver_type} for _, review, deliver_type in coupang_reviews if _ == star]} for star in range(1, 6)]
    # Coupang 별점 분포 데이터 가져오기
    coupang_star_counts = db.query(CoupangReview.star, func.count(CoupangReview.star)).group_by(CoupangReview.star).all()
    coupang_star_result = [{"star": star, "count": count} for star, count in coupang_star_counts]
    db.close()
    return {"coupang_reviews": coupang_result, "coupang_star_counts": coupang_star_result}


@app.get("/data/analysis", response_class=HTMLResponse)
async def read_data():
    session = SessionLocal()
    try:
        # Naver 리뷰와 Coupang 리뷰 데이터 로드
        naver_reviews = session.query(NaverReview).all()
        coupang_reviews = session.query(CoupangReview).all()

        # pandas DataFrame으로 변환
        naver_df = pd.DataFrame([r.__dict__ for r in naver_reviews])
        coupang_df = pd.DataFrame([r.__dict__ for r in coupang_reviews])

        # 필요한 열만 선택
        required_columns = ['review', 'Predict_segment']
        naver_df = naver_df[required_columns]
        coupang_df = coupang_df[required_columns]

        # 두 데이터프레임 결합
        combined_df = pd.concat([naver_df, coupang_df])

        # -1을 'Negative'로, 1을 'Positive'로 변경
        combined_df['sentiment'] = combined_df['Predict_segment'].apply(lambda x: 'Negative' if x == -1 else 'Positive')

        # 감정 개수 계산
        sentiment_counts = combined_df['sentiment'].value_counts()
        sentiment_ratios = combined_df['sentiment'].value_counts(normalize=True) * 100

        # 감정 분포 시각화 (막대 그래프)
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
        ax.set_title('Sentiment Analysis')
        ax.set_ylabel('Count')
        ax.set_xlabel('Sentiment')
        ax.set_xticklabels(sentiment_counts.index, rotation=0)

        # 막대 그래프를 HTML로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        bar_img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 감정 비율 시각화 (파이 차트)
        fig, ax = plt.subplots()
        ax.pie(sentiment_ratios, labels=sentiment_ratios.index, autopct='%1.1f%%', colors=['skyblue', 'orange'])
        ax.set_title('Sentiment Ratio')

        # 파이 차트를 HTML로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pie_img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 긍정 리뷰 워드클라우드 생성
        positive_reviews = combined_df[combined_df['sentiment'] == 'Positive']['review'].str.cat(sep=' ')
        positive_wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(positive_reviews)
        
        fig, ax = plt.subplots()
        ax.imshow(positive_wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Positive Reviews Wordcloud')

        # 긍정 워드클라우드를 HTML로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        positive_wc_img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 부정 리뷰 워드클라우드 생성
        negative_reviews = combined_df[combined_df['sentiment'] == 'Negative']['review'].str.cat(sep=' ')
        negative_wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(negative_reviews)
        
        fig, ax = plt.subplots()
        ax.imshow(negative_wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Negative Reviews Wordcloud')

        # 부정 워드클라우드를 HTML로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        negative_wc_img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # HTML 내용 생성
        html_content = f"""
        <html>
        <head>
        <title>Sentiment Analysis</title>
        </head>
        <body>
        <h1>Sentiment Analysis</h1>
        <h2>Count</h2>
        <img src="data:image/png;base64,{bar_img_str}" alt="Sentiment Analysis">
        <h2>Ratio</h2>
        <img src="data:image/png;base64,{pie_img_str}" alt="Sentiment Ratio">
        <h2>Positive Reviews Wordcloud</h2>
        <img src="data:image/png;base64,{positive_wc_img_str}" alt="Positive Reviews Wordcloud">
        <h2>Negative Reviews Wordcloud</h2>
        <img src="data:image/png;base64,{negative_wc_img_str}" alt="Negative Reviews Wordcloud">
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"데이터 분석에 실패했습니다: {str(e)}")
    finally:
        session.close()

product_groups = {
    "콘드로이친": ["C1300013", "C1300016", "C1300018"],
    "우슬": ["C1300003", "C1300004"],
    "보스웰리아": ["C1300001", "C1300002", "C1300015", "C1300017", "C1300019"],
    "가자": ["C1300005"],
    "msm": ["C1300007", "C1300008", "C1300025", "C1300026"],
    "칼마디": ["C1300023", "C1300024"],
    "비타민": ["C1300022"],
    "콜라겐": ["C1300020"],
    "크릴오일": ["C1300021"],
    "NAG": ["C1300009"]
}

def generate_wordcloud(reviews, title, font_path):
    if reviews.empty:
        return ""
    
    reviews_text = reviews.str.cat(sep=' ')
    wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(reviews_text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str

from sqlalchemy.orm import Session

@app.get("/data/analysis/quarter", response_class=HTMLResponse)
def analyze_reviews_by_quarter():
    session = SessionLocal()
    
    try:
        # Retrieve all reviews from both Naver and Coupang
        naver_reviews = session.query(NaverReview).all()
        coupang_reviews = session.query(CoupangReview).all()

        # Convert reviews to pandas DataFrame
        naver_df = pd.DataFrame([r.__dict__ for r in naver_reviews])
        coupang_df = pd.DataFrame([r.__dict__ for r in coupang_reviews])

        # Concatenate the two DataFrames
        combined_df = pd.concat([naver_df, coupang_df])

        # Prepare data for analysis
        combined_df['review_date'] = pd.to_datetime(combined_df['review_date'])
        combined_df['quarter'] = combined_df['review_date'].dt.to_period('Q')
        combined_df['sentiment'] = combined_df['Predict_segment'].apply(lambda x: 'Negative' if x == -1 else 'Positive')

        # Group by quarter and sentiment
        grouped = combined_df.groupby(['quarter', 'sentiment']).size().unstack().fillna(0)

        # HTML content to return
        html_content = "<html><head><title>Review Analysis by Quarter</title></head><body><h1>Review Analysis by Quarter</h1>"

        for quarter in grouped.index:
            # Data for the current quarter
            quarter_data = grouped.loc[quarter]

            # Plot histogram for the current quarter
            fig, ax = plt.subplots()
            quarter_data.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
            ax.set_title(f'Review Count in {quarter}')
            ax.set_ylabel('Count')
            ax.set_xlabel('Sentiment')
            plt.xticks(rotation=0)

            # Convert the histogram to a base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            hist_img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            # Plot pie chart for the current quarter
            fig, ax = plt.subplots()
            quarter_data.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['skyblue', 'orange'])
            ax.set_title(f'Sentiment Ratio in {quarter}')
            ax.set_ylabel('')

            # Convert the pie chart to a base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            pie_img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            # Append charts to HTML content
            html_content += f"<h2>{quarter}</h2>"
            html_content += f"<h3>Review Count</h3><img src='data:image/png;base64,{hist_img_str}' alt='Review Count in {quarter}'>"
            html_content += f"<h3>Sentiment Ratio</h3><img src='data:image/png;base64,{pie_img_str}' alt='Sentiment Ratio in {quarter}'>"

        html_content += "</body></html>"

        return HTMLResponse(content=html_content)
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to analyze reviews: {str(e)}")
    finally:
        session.close()

@app.get("/data/analysis/product", response_class=HTMLResponse)
def analyze_product_reviews(product_codes: List[str] = Query(..., description="List of product codes to analyze")):
    session = SessionLocal()
    
    try:
        # Naver 리뷰와 Coupang 리뷰 데이터 로드
        naver_reviews = session.query(NaverReview).filter(NaverReview.product_code.in_(product_codes)).all()
        coupang_reviews = session.query(CoupangReview).filter(CoupangReview.product_code.in_(product_codes)).all()

        # pandas DataFrame으로 변환
        naver_df = pd.DataFrame([r.__dict__ for r in naver_reviews])
        coupang_df = pd.DataFrame([r.__dict__ for r in coupang_reviews])

        # 필요한 열만 선택
        required_columns = ['review', 'Predict_segment']
        naver_df = naver_df[required_columns]
        coupang_df = coupang_df[required_columns]

        # 두 데이터프레임 결합
        combined_df = pd.concat([naver_df, coupang_df])

        # -1을 'Negative'로, 1을 'Positive'로 변경
        combined_df['sentiment'] = combined_df['Predict_segment'].apply(lambda x: 'Negative' if x == -1 else 'Positive')

        # 감정 개수 계산
        sentiment_counts = combined_df['sentiment'].value_counts()
        sentiment_ratios = combined_df['sentiment'].value_counts(normalize=True) * 100

        # 감정 분포 시각화 (막대 그래프)
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
        ax.set_title('Sentiment Analysis')
        ax.set_ylabel('Count')
        ax.set_xlabel('Sentiment')
        ax.set_xticklabels(sentiment_counts.index, rotation=0)

        # 막대 그래프를 HTML로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        bar_img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 감정 비율 시각화 (파이 차트)
        fig, ax = plt.subplots()
        ax.pie(sentiment_ratios, labels=sentiment_ratios.index, autopct='%1.1f%%', colors=['skyblue', 'orange'])
        ax.set_title('Sentiment Ratio')

        # 파이 차트를 HTML로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pie_img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 긍정 리뷰 워드클라우드 생성
        positive_reviews = combined_df[combined_df['sentiment'] == 'Positive']['review']
        positive_wc_img_str = generate_wordcloud(positive_reviews, 'Positive Reviews Wordcloud', font_path)

        # 부정 리뷰 워드클라우드 생성
        negative_reviews = combined_df[combined_df['sentiment'] == 'Negative']['review']
        negative_wc_img_str = generate_wordcloud(negative_reviews, 'Negative Reviews Wordcloud', font_path)

        # HTML 내용 생성
        html_content = f"""
        <html>
        <head>
        <title>Sentiment Analysis</title>
        </head>
        <body>
        <h1>Sentiment Analysis</h1>
        <h2>Count</h2>
        <img src="data:image/png;base64,{bar_img_str}" alt="Sentiment Analysis">
        <h2>Ratio</h2>
        <img src="data:image/png;base64,{pie_img_str}" alt="Sentiment Ratio">
        """
        
        if positive_wc_img_str:
            html_content += f"""
            <h2>Positive Reviews Wordcloud</h2>
            <img src="data:image/png;base64,{positive_wc_img_str}" alt="Positive Reviews Wordcloud">
            """
        
        if negative_wc_img_str:
            html_content += f"""
            <h2>Negative Reviews Wordcloud</h2>
            <img src="data:image/png;base64,{negative_wc_img_str}" alt="Negative Reviews Wordcloud">
            """

        html_content += """
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"데이터 분석에 실패했습니다: {str(e)}")
    finally:
        session.close()



@app.get("/data/analysis/{product_name}", response_class=HTMLResponse)
def analyze_specific_product_reviews(product_name: str):
    if product_name not in product_groups:
        raise HTTPException(status_code=400, detail="Invalid product name")
    product_codes = product_groups[product_name]
    return analyze_product_reviews(product_codes=product_codes)

