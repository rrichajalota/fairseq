from easynmt import EasyNMT
m2pivot = EasyNMT('m2m_100_418M') # mbart50_en2m
pivot2m = EasyNMT('m2m_100_418M') # mbart50_m2en

# process_pool = en2es.start_multi_process_pool(['cuda', 'cuda'])
# pp = es2en.start_multi_process_pool(['cuda', 'cuda'])
# /netscratch/jalota/datasets/haifa-hansard/dev/original.txt
with open("/netscratch/jalota/datasets/motra-preprocessed/de_all/train/og_bal.tok") as f:
  lines = f.readlines()
  # es_translations = en2es.translate(lines, target_lang='es')
  count = 0
  es_trans = []
  with open("/netscratch/jalota/datasets/motra-preprocessed/de_all/train/de2en.txt", "w") as fo:
    for translation in m2pivot.translate_stream(lines, show_progress_bar=True, chunk_size=80, source_lang='de', target_lang='en'):
      count +=1 
      print(count)
      # es_trans.append(translation)
      fo.write(translation)
      # fo.write("\n")
  #Do some warm-up
  # en2es.translate_multi_process(process_pool, lines[:100], source_lang='en', target_lang='es', show_progress_bar=False)
  # es_translations_multi_p = en2es.translate_multi_process(process_pool, lines, source_lang='en', target_lang='es', show_progress_bar=True)
  # en2es.stop_multi_process_pool(process_pool)

  #Do some warm-up
  # es2en.translate_multi_process(pp, es_translations_multi_p[:100], source_lang='es', target_lang='en', show_progress_bar=False)
  # en_translations_multi_p = es2en.translate_multi_process(pp, es_translations_multi_p, source_lang='es', target_lang='en', show_progress_bar=True)
  # es2en.stop_multi_process_pool(pp)
  # en_trans = []
with open("/netscratch/jalota/datasets/motra-preprocessed/de_all/train/de2en.txt") as f:
  lines = f.readlines()
  count = 0
  with open("/netscratch/jalota/datasets/motra-preprocessed/de_all/train/bt_de.txt", "w") as foo:
    for translation in pivot2m.translate_stream(lines, show_progress_bar=True, chunk_size=80, source_lang='en', target_lang='de'):
      count +=1 
      print(count)
      foo.write(translation)
      # foo.write("\n")
      # en_trans.append(translation)
    # en_translations = es2en.translate(es_trans, target_lang='en')

# --- Remove blank lines ---
# awk -i inplace NF bt_og.txt 

# with open("/netscratch/jalota/datasets/motra-preprocessed/en_es/test/bt_og.txt", "w") as f:
#   for tr in en_trans:    
#     f.write(tr)
#     f.write("\n")