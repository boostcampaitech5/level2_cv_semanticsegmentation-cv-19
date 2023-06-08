# 모델 추가시 규칙
만든 모델의 이름이 UNet일때
1) file의 name은 unet_custom.py로 하고 (소문자_custom.py 형식)
2) Class의 name 은 UNet으로 한다. (첫 글자는 대문자로 하자)
3) config.json에는 Class의 이름을 넣는다. (ex. "model": "UNet")