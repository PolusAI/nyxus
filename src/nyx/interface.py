import sys
#XXXXXXXX	import backend

import os
TestData = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testdata.bin")

def is_nyx_alive():
	print ("is_nyx_alive responded")
	return True
	
def is_nyx_alive2(arg):
	print ("is_nyx_alive2 received " + str(arg))
	return True	
	
