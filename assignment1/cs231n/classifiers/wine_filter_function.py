# 와인 필터링 추천 함수 선언

def MyWines(food, price, rate, celeb):
  import pandas as pd
  wines = pd.read_csv("wines_final2.csv")
  mask1 = (wines.Rating >= rate) & (wines.Price2 == price)
  df_mywines1 = wines.loc[mask1,:]
  mask2 =  (df_mywines1.Celeb == celeb)
  df_mywines2 = df_mywines1.loc[mask2,:]
  if food == "해산물":
    mask3 = (df_mywines2.Type == 'White') | (df_mywines2.Type == 'Rose')
  elif food == "식전주":
    mask3 = (df_mywines2.Type == 'White')
  elif food == "과일":
    mask3 = (df_mywines2.Type == 'Sparkling')
  elif food == "육류":
    mask3 = (df_mywines2.Type == 'Red')
  elif food == "기타":
    mask3 = (df_mywines2.Type == 'White') | (df_mywines2.Type == 'Rose') | (df_mywines2.Type == 'Red') | (df_mywines2.Type == 'Sparkling')
  df_mywines3 = df_mywines2.loc[mask3,:]
  df_mywines3.sort_values(by='Price(\)')
  mywine5 = df_mywines3[:5]
  mywine5 = mywine5[['Name','Country','Rating','Price(\)','Type']]
  mywine5 = mywine5.sort_values(by='Price(\)',ignore_index=True)
  out = "[당신을 위한 와인 Top5]\n"
  for i in range(len(mywine5)):
    out += "\n"+str(i+1)+". [{}]{} ({}): 레이팅 {}점, 가격 {}만원입니다.\n".format(mywine5['Type'][i],mywine5['Name'][i],mywine5['Country'][i],mywine5['Rating'][i],mywine5['Price(\)'][i])
  return out

##main##
price = '5~6만원대'
rate = 3
food = '해산물'
celeb = 'no'
a = MyWines(food, price, rate, celeb)
print(a)