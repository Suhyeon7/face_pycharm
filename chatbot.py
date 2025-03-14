## https://console.picovoice.ai/
import json

import requests
import xmltodict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os, sys
from urllib import request
from bs4 import BeautifulSoup
import asyncio
import datetime
#from chatGPT import callChatGPT  # 직접 만든 ChatGPT 호출 모듈
from pydub import AudioSegment
from pydub.playback import play
from fastapi.middleware.cors import CORSMiddleware


try:
    import speech_recognition as sr
    import pvporcupine
    from pvrecorder import PvRecorder
    from gtts import gTTS
    from playsound import playsound
    import feedparser
except ImportError:
    os.system('pip install --upgrade pip')
    os.system('pip install SpeechRecognition')
    os.system('pip install pvporcupine')
    os.system('pip install pvrecorder')
    os.system('pip install gtts')
    os.system('pip install feedparser')
    os.system('pip install playsound==1.2.2')
    sys.exit()

print("[대기] 잠시만 기다려주세요..")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React 개발 서버를 허용. 필요시 특정 도메인만 추가
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 데이터 저장소 (간단한 글로벌 상태)
ai_responses = []

async def my_tts(text):
    print('[AI] : ' + text)
    ai_responses.append(text)
    print(ai_responses)
    tts = gTTS(text=text, lang='ko')

    # 임시 음성 파일 경로 설정
    file_name = 'voice.mp3'
    file_path = os.path.abspath(file_name)

    # 음성 파일 저장
    tts.save(file_path)


    # 비동기적으로 재생
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, playsound, file_path)


    # 재생 후 파일 삭제
    if os.path.exists(file_path):
        os.remove(file_path)

@app.get("/chatbot")
async def get_ai_response():
    print("AI Responses:", ai_responses)
    return JSONResponse(content={"responses": ai_responses[-1:]})
    #return JSONResponse(content={"responses": ["No response yet"]})



# 예시: STT 함수 내부에서 비동기로 my_tts 호출
async def STT():
    r = sr.Recognizer()
    with sr.Microphone(0) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='ko-KR')
            print('You said: {}'.format(text))
            await ai(text)  # ai 함수에서 my_tts를 호출할 때는 await 사용
        except Exception as err:
            print(err)
            await my_tts("잘 못 들었어요. 다시 말해주세요.")
            await STT()


### 뉴스 ####
def my_news():
    url = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
    news_data = []
    news_rss = feedparser.parse(url)
    for title in news_rss.entries:
        news_data.append(title.title)
    return news_data

#####버스정보조회###################
def my_bus():
    serviceKey = '%2Fp%2BOfttfs7VZOhAT1rMk6X0crpnWbLU6MJ3C8DZ7aenZbMH5jEWReRIhm9UfwAtRrgw0bEZqjnkF471tHl%2ByoQ%3D%3D'
    # 정류소 id
    # stationId="115000302"
    stationId = "108000019"
    # 버스 ID
    url = "http://ws.bus.go.kr/api/rest/arrive/getLowArrInfoByStId?serviceKey={}&stId={}".format(serviceKey, stationId)
    # get으로 요청함
    response = requests.get(url).content
    # xml파일을 dict로 파싱하여 사용
    dict = xmltodict.parse(response)

    # 원하는 데이터가 ServiceResult 내부 msgBody 내부 itemList내부에 있음
    # 다시 dict로 받은 값을 Json로 변환
    jsonString = json.dumps(dict['ServiceResult']['msgBody']['itemList'], ensure_ascii=False)
    # json을 형태로 받은 데이터를 Python의 객체로 변환 (dict)
    jsonObj = json.loads(jsonString)

    msg = ''
    for i in range(len(jsonObj)):
        msg += '{}\n  {}\n 다음 버스: {}\n'.format(jsonObj[i]['rtNm'], jsonObj[i]['arrmsg1'], jsonObj[i]['arrmsg2'])
    return msg
### 날씨 ###
def weather_info():
    url_weather = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=109"
    html = request.urlopen(url_weather)

    soup = BeautifulSoup(html, 'lxml')
    # location 태그를 찾습니다.
    output = ""
    for location in soup.find_all("location"):
        # 내부의 city, wf, tmn, tmx 태그를 찾아 출력합니다.
        #city = location.find("city").string
        city = "서울"
        weather = location.find("wf").string
        min = location.find("tmn").string
        max = location.find("tmx").string

    # 미세먼지
    today = "오늘 {0} 날씨입니다. 오늘 날씨는 {1}이고, 최고 온도는 {2}도, 최저 온도는 {3}도 입니다.".format(city, weather, max, min)
    today = today + "오늘의 날씨 정보였습니다."
    return today


### ChatGPT 답변을 위한 코드 #########################################
async def baby(speech):
    future1 = asyncio.ensure_future(my_tts("인공지능이 답변을 생성중입니다. 시간이 걸릴 수 있으니, 잠시 대기해주세요."))
    #future2 = asyncio.ensure_future(callChatGPT(speech))

    #await asyncio.gather(future1, future2)

    #await my_tts(future2.result())


### 판단 #############################################################
async def ai(speech):  #
    if '뉴스' in speech:
        texts = my_news()
        await my_tts('오늘 주요 뉴스입니다.')
        for text in texts[0:3]:
            await my_tts(text)

    elif '날씨' in speech:
        #await my_tts("현재 날씨는 준비중이에요.")
        await my_tts(weather_info())

    elif '뭐야' in speech:
        await baby(speech)

    elif '종료' in speech:
        await my_tts("다음에 또 만나요")

    elif '날짜' in speech or '며칠' in speech:
        now = datetime.datetime.now()
        await my_tts("오늘은 "+now.strftime("%Y년 %m월 %d일")+"입니다")

    elif '시간' in speech or '몇 시' in speech:
        now = datetime.datetime.now()
        await my_tts("지금은 " + now.strftime("%H시 %M분 %S초") + "입니다")

    elif '버스' in speech:
        await my_tts(my_bus())
    else:
        await my_tts("다시 한번 말씀해주세요.")
        await STT()


## Wake word Setting
porcupine = pvporcupine.create(
    access_key='zsOSJvn7kIUYEj4FG/G9lMWVXHGE2i4s9q2ig3KIwak4PD6kBz7+Sw==',
    keyword_paths=["model/하이아이엠_ko_mac_v3_0_0.ppn"],  # 다운 받은 키워드 파일, 여러개로도 구성 가능하다.
    model_path="model/porcupine_params_ko.pv",  # Github에서 다운 받은 파일
)


# Porcupine으로 키워드 감지 후 STT 함수 비동기 실행
async def detect_keyword():
    ## Mic Setting (device_index 변경해야함)
    # 사용 가능한 오디오 장치 출력
    devices = PvRecorder.get_available_devices()
    print("Available devices:")
    for i, device in enumerate(devices):
        print(f"[{i}] {device}")
    recorder = PvRecorder(frame_length=512, device_index=0)
    recorder.start()
    while True:
        pcm = recorder.read()
        keyword_index = porcupine.process(pcm)
        if keyword_index == 0:
            recorder.delete()
            await my_tts("무엇을 도와드릴까요?")
            await STT()
            recorder = PvRecorder(frame_length=512, device_index=0)
            recorder.start()


# 비동기 루프 시작
async def main():
    # 임시 음성 파일 경로 설정
    file_name = 'turn_on.mp3'
    file_path = os.path.abspath(file_name)

    # 비동기적으로 재생
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, playsound, file_path)
    print("[실행 가능] 마이크가 준비되었습니다")

    await detect_keyword()


# 비동기 루프 실행
if __name__ == '__main__':
    asyncio.run(main())