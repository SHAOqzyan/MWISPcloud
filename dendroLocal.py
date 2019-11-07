


from astrodendro import Dendrogram

from astropy.io import fits
from matplotlib import pyplot as plt
from astropy import units as u
import numpy as np

from astropy.table import Table
from astrodendro import Dendrogram, ppv_catalog

import datetime
from myTree import dendroTree

# meta data of 12CO




class MWcloud:



	metadata = {}
	metadata['data_unit'] = u.K
	metadata['spatial_scale'] =  0.5 * u.arcmin
	metadata['beam_major'] =  50/60. * u.arcmin # FWHM
	metadata['beam_minor'] =  50/60. * u.arcmin # FWHM
	metadata['velocity_scale'] =  0.2 * u.km/u.s # FWHM

	c= 299792458.
	f=115271202000.0
	wavelength=c/f*u.meter

	metadata['wavelength'] = wavelength  # 22.9 * u.arcsec # FWHM



	CO12FITS=None

	def __init__(self ,regionName ="G2650"):
		self.regionName=regionName


	def doDendro(self, COFITS,minV=3 , minPix=8,RMS=0.5,minDelta=3,saveName=None ):
		"""
		COFITS may not be masedk
		:param COFITS:
		:param minV:
		:param minPix:
		:param RMS:
		:param minDelta:
		:return:
		"""
		saveMark="minV{}minP{}".format(minV,minPix)

		treeFile= saveMark+"_Tree.txt"

		catFile=  saveMark + "_dendroCat.fit"
		dendroFile=  saveMark + "_dendro.fits"

		trunkCatFile=  saveMark + "_dendroCatTrunk.fit"
		trunkFITS=  saveMark + "_TrunkAsign.fits"



		hdu= fits.open(COFITS)[0]

		dataCO=hdu.data
		headCO=hdu.header

		#mask the data by minValue
		print "Cmputing dendrogram with min value {}, min pixel number {}".format(minV*RMS,minPix)
		#data[data<minV*RMS]=0
		d = Dendrogram.compute(dataCO, min_value=minV*RMS,  verbose=1, min_npix=minPix, min_delta=minDelta*RMS )

		self.produceAssignFITS(d,COFITS,trunkFITS)

		if saveName!=None:
			d.save_to( "localDendro.fits"  )


		else:
			d.save_to( dendroFile )

		#calculate the catalog
		cat = ppv_catalog(d, self.metadata)

		cat.write(catFile, overwrite=True)
		self.writeTreeStructure(d,treeFile)

		doTree=dendroTree(treeFile)
		trunks=doTree.getAllTrunk()

		trunkCat=cat[ trunks]
		print "{} trunks found!".format(len(trunkCat))
		trunkCat.write(trunkCatFile, overwrite=True)



	def produceAssignFITS(self, d, COFITS,  saveFITS  ):
		"""
		#only consider trunks
		:param d:
		:param saveFITS:
		:return:
		"""
		hdu= fits.open(COFITS)[0]

		data=hdu.data
		head=hdu.header

		data0=np.zeros_like(data)



		for eachT in  d.trunk:

			tID =   eachT.idx
			#print eachT.parent

			tMask=eachT.get_mask().astype("short")

			data0= data0 + tMask*tID


		fits.writeto(saveFITS, data0, header=head,  overwrite=True  )






	def writeTreeStructure(self,dendro,saveName):

		f=open( saveName,'w')

		#for eachC in self.dendroData:
		for eachC in dendro:

			parentID=-1

			p=eachC.parent

			if p!=None:

				parentID=p.idx

			fileRow="{} {}".format(eachC.idx,parentID)
			f.write(fileRow+" \n")

		f.close()





doCloud=MWcloud()
doCloud.doDendro("G2650Local30.fits",minV=3,minPix=8)
doCloud.doDendro("G2650Local30.fits",minV=2,minPix=16)

