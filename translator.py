# translate_helper.py

import pandas as pd
from deep_translator import GoogleTranslator

def get_language_options():
    return {
        'French': 'fr',
        'Spanish': 'es',
        'German': 'de',
        'Tamil': 'ta',
        'Hindi': 'hi',
        'Japanese': 'ja',
        'Chinese (Simplified)': 'zh-CN',
        'Arabic': 'ar',
        'Russian': 'ru',
        'Malayalam': 'ml',
    }

def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        return f"❌ Translation failed: {e}"
