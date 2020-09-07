##
python preprocess.py  --destdir Comparable/data-bin/en_ne_sp6k/ \
  --source-lang src \
  --target-lang tgt \
  --trainpref /path/to/corpus.{src|tgt}  \  
  --validpref /path/to/valid.{src|tgt}  \
  --dataset-impl raw \
  --joined-dictionary \
  --workers 60  
