from utils.diskmap_utils import dump_to_bytes, write_xobject_to_bin_file
import torch
import tqdm
def convert_dict(d, outpath):
    xbin = open(outpath, "wb")
    xbin_aux = open("%s.aux" % outpath, "w")
    sz0 = 0
    for k, v in tqdm.tqdm(d.items(), total=len(d)):
        v = dump_to_bytes(v)
        sz = write_xobject_to_bin_file(v, xbin, bin_obj=True)
        xbin_aux.write("%s\t%s\n" % (k, sz0))
        sz0 += sz
    xbin.close()
    xbin_aux.close()


def run_convert():
    path = "/home/gpux1/Codes/DiffDock/data/esm2_3billion_embeddings.pt"
    outpath = "%s.dxobj" % path

    d = torch.load(path)

    convert_dict(d, outpath)


if __name__ == "__main__":
    run_convert()