# flowers102
 
all code should be functional from the classifer.ipynb file
not including the dataset in the repo for obvious reasons

values for dataloaders etc are chosen based on how easily it would run on a
GTX 2060 gpu. The training time will increase massively on the occasion that
you load more data than you have dedicated gpu VRAM as the preloaded pin 
memory from ram has to be sent to the gpu first