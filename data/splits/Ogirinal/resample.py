import random
import os
import glob
random.seed(1)
def sample_from_file(path, r = .2):
	px = "%s_origin" % path
	if not os.path.exists(px):
		os.system("mv \"%s\" \"%s\"" % (path, px))
	with open(px) as f:
		lines = f.readlines()
	n = int(r * len(lines))
	s_lines = random.sample(lines, n)
	with open(path, "w") as fout:
		for l in s_lines:
			fout.write("%s" % l)
		fout.close()

def do_sample():
	ps = glob.glob("*timesplit*")
	for p in ps:
		sample_from_file(p, 0.2)
if __name__ == "__main__":
	do_sample()

