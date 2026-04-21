import os
from PIL import Image, ImageDraw
import requests

os.makedirs('data', exist_ok=True)
img_path = os.path.join('data', 'sample_label.png')
text = 'Ingredients: wheat flour, sugar, peanut oil, milk powder'
img = Image.new('RGB', (1400, 220), color='white')
ImageDraw.Draw(img).text((20, 80), text, fill='black')
img.save(img_path)
print(f'CREATED_IMAGE={img_path}')

try:
    with open(img_path, 'rb') as f:
        resp_post = requests.post('http://127.0.0.1:8000/analyze-image', files={'file': ('sample_label.png', f, 'image/png')}, timeout=20)
    print(f'POST_STATUS={resp_post.status_code}')
    try:
        pj = resp_post.json()
    except Exception:
        pj = {'raw': resp_post.text[:300]}
    ingredients = None
    if isinstance(pj, dict):
        for key in ['ingredients','ingredient_list','detected_ingredients','parsed_ingredients','items']:
            if key in pj:
                ingredients = pj[key]
                break
    print(f'POST_INGREDIENTS_SNIPPET={str(ingredients)[:300] if ingredients is not None else str(pj)[:300]}')
except Exception as e:
    print('POST_STATUS=ERROR')
    print(f'POST_INGREDIENTS_SNIPPET={e}')

try:
    params = {'vhi':0.65,'cas':0.5,'erf':0.55,'ingredients':'wheat flour,sugar,peanut oil,milk powder'}
    resp_get = requests.get('http://127.0.0.1:8000/explain', params=params, timeout=20)
    print(f'GET_STATUS={resp_get.status_code}')
    try:
        gj = resp_get.json()
    except Exception:
        gj = {'raw': resp_get.text[:400]}
    top3 = None
    if isinstance(gj, dict):
        if isinstance(gj.get('top_features'), list):
            top3 = gj['top_features'][:3]
        elif isinstance(gj.get('feature_importance'), dict):
            top3 = sorted(gj['feature_importance'].items(), key=lambda kv: kv[1], reverse=True)[:3]
        elif isinstance(gj.get('feature_importance'), list):
            top3 = gj['feature_importance'][:3]
        elif isinstance(gj.get('importance'), dict):
            top3 = sorted(gj['importance'].items(), key=lambda kv: kv[1], reverse=True)[:3]
        elif isinstance(gj.get('importance'), list):
            top3 = gj['importance'][:3]
    print(f'GET_TOP3_FEATURES_SNIPPET={str(top3)[:400] if top3 is not None else str(gj)[:400]}')
except Exception as e:
    print('GET_STATUS=ERROR')
    print(f'GET_TOP3_FEATURES_SNIPPET={e}')
