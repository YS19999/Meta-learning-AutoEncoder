import datetime
from embedding.wordebd import WORDEBD
from embedding.bert import CXTEBD

from model.discriminator import Discriminator
from model.encoder import Encoder
from model.decoder import Decoder

def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.bert:
        ebd = CXTEBD(args)
    else:
        ebd = WORDEBD(vocab, args.finetune_ebd)

    encoder = Encoder(ebd, args)
    decoder = Decoder(encoder.ebd_dim, encoder.ebd_embedding, args)
    modelD = Discriminator(args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        encoder = encoder.cuda(args.cuda)
        decoder = decoder.cuda(args.cuda)
        modelD = modelD.cuda(args.cuda)
        return encoder, decoder, modelD
    else:
        return encoder, decoder, modelD