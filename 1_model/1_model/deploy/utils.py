import pandas as pd
import numpy as np

def wrangle(data, test=False):
    """
    ทำความสะอาดและแปลงข้อมูล (Data Preprocessing)
    Parameters:
        data (dict, list, หรือ DataFrame): ข้อมูลที่ต้องการประมวลผล
        test (bool): ถ้าเป็น True จะใช้ logic สำหรับ test data เช่น fillna
    Returns:
        DataFrame ที่พร้อมสำหรับการนำไปใช้กับโมเดล
    """
    # ถ้า data เป็น dict หรือ list ให้แปลงเป็น DataFrame ก่อน
    if isinstance(data, (dict, list)):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # ✅ ตัวอย่างการจัดการ missing values
    df = df.fillna(0)

    # ✅ ตัวอย่างการ encode categorical variables
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    # ✅ แยก logic กรณี test=True
    if test:
        print("Running in TEST mode - minimal cleaning applied")

    return df