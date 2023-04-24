import torch

de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model',
                       tokenizer='moses', bpe='fastbpe')

de2en.eval()
en2de.eval()

en2de.cuda()
de2en.cuda()

with open("/netscratch/jalota/datasets/motra-preprocessed/en_de/train/original.txt") as f:
  lines = f.readlines()
  out = []
  de = en2de.translate(lines)
  print("done de translations")
  bt = de2en.translate(de)
  print("got round-trip translations")
  # out.append(bt)

  with open("/netscratch/jalota/datasets/motra-preprocessed/en_de/train/bt_original.txt", "w") as foo:
      for line in bt:
          foo.write(line)
          foo.write("\n")