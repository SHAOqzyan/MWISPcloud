


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
from scimes import SpectralCloudstering



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


	rms=0.5

	myVolume='trueVolume'
	myVrms='trueVrms'

	myRatio='trueRatio' #bad

	myFlux='trueFluxRatio' #residual flux of two chidren
	lostFluxRatio= 'lostFluxRatio' # lost of flux except of al leaves




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


	def getTrueVolumeAndVrms(self,treeFile,cat,subRegion=''):
		"""

		:param d:
		:param cat:
		:return:
		"""

		# calculate true volumes and true Vrms, by Vrms, we mean the velocity different of two leaves,which is more reasonable

		# by true volumes we mean all the sum of leaves? ,better the former


		#first get Tree


		doTree= dendroTree(treeFile,dendroCat=cat)

		#doTree.getAllLeaves(0)

		dendroTB=  doTree.dendroTB

		dendroTB[self.myVolume  ] = dendroTB["flux"]
		dendroTB[self.myVrms ] = dendroTB["v_rms"]

		dendroTB[self.myRatio ] = dendroTB["v_rms"]
		dendroTB[self.myFlux ] = dendroTB["v_rms"]
		dendroTB[self.lostFluxRatio ] = dendroTB["v_rms"]


		for eachRow in   dendroTB:
			#allL=doTree.getAllLeaves(eachRow["_idx"] )
			cID= eachRow["_idx"]

			twoChildren=doTree.getChildren(  cID )





			if twoChildren==None :

				eachRow[self.myVolume ] =  eachRow["area_exact"]* eachRow["v_rms"]    #eachRow["v_rms"]

				eachRow[self.myVrms] = eachRow["v_rms"]  #doTree.getMaxDV(cID)
				eachRow[self.myRatio] =1 #doTree.getMaxDV(cID)
				#eachRow[self.myFlux] =1  #doTree.getMaxDV(cID)

				eachRow[self.myFlux] =0  #doTree.getMaxDV(cID)

				eachRow[self.lostFluxRatio] =0  #doTree.getMaxDV(cID)

			else:

				eachRow[self.myVrms],lostFlux,nleaves = doTree.getMaxDVAnLostFlux(cID)
				eachRow[self.myVolume] =   eachRow["area_exact"]*eachRow[self.myVrms]    #dendroTB[twoChildren[0]]["area_exact"]*dendroTB[twoChildren[0]]["v_rms"]+  dendroTB[twoChildren[1]]["area_exact"]*  dendroTB[twoChildren[1]]["v_rms"]

				eachRow[self.myRatio] =  eachRow["area_exact"]/ (dendroTB[twoChildren[0]]["area_exact"] +  dendroTB[twoChildren[1]]["area_exact"])

				#eachRow[self.myFlux] =  eachRow["flux"]/ (dendroTB[twoChildren[0]]["flux"] +  dendroTB[twoChildren[1]]["flux"])
				eachRow[self.myFlux] =  (dendroTB[twoChildren[0]]["flux"] +  dendroTB[twoChildren[1]]["flux"])/eachRow["flux"]
				eachRow[self.lostFluxRatio]=1-lostFlux/eachRow["flux"]



		return dendroTB[self.myVolume ], dendroTB[self.myVrms], dendroTB[self.myRatio], dendroTB[self.myFlux],dendroTB[self.lostFluxRatio]





	def doSCIMES(self,CO12FITS, dendroFITS,catName,saveMarker,criteriaUsed=None,scales=None,saveAll=True,inputD=None):

		#since reading dendrogram
		if inputD==None:
			d=Dendrogram.load_from( dendroFITS )
		else:
			d=inputD


		cat = Table.read(catName)


		hdu= fits.open(CO12FITS)[0]

		dataCO=hdu.data
		headCO=hdu.header


		print ""
		print len(cat),"<-- total number of structures?"
		treeFile="tmpTree{}.txt".format(saveMarker)
		self.writeTreeStructure(d,treeFile)

		newVo,newVe,newRatio, newFlux, lostFlux = self.getTrueVolumeAndVrms(treeFile,catName )
		cat[self.myVolume]=newVo
		cat[self.myVrms]=newVe
		cat[self.myRatio]=newRatio
		cat[self.myFlux]=newFlux
		cat[self.lostFluxRatio ] = lostFlux



		print "Processing clustering..."
		if criteriaUsed!=None and scales!=None:
			res = SpectralCloudstering(d, cat, headCO, criteria = criteriaUsed  , user_scalpars=scales,  blind = True , rms = self.rms, s2nlim = 3, save_all = saveAll, user_iter=1)
		else:
			res = SpectralCloudstering(d, cat, headCO, criteria = ["volume","luminosity"]   ,  blind = True , rms = self.rms, s2nlim = 3, save_all = saveAll, user_iter=1)

		print "Done..."


		res.clusters_asgn.writeto ( 'ClusterAsgn_{}.fits'.format(saveMarker), overwrite=True)

		clusterCat=cat[res.clusters]

		clusterCat.write(  "ClusterCat_{}.fit".format(saveMarker), overwrite=True)
		cat.write(  "DendroCatExtended_{}.fit".format(saveMarker), overwrite=True)#same as the catlog of dendrogram, but with several extended columns. used to see how good the criterin




doCloud=MWcloud()
#doCloud.doDendro("G2650Local30.fits",minV=3,minPix=8)
#doCloud.doDendro("G2650Local30.fits",minV=2,minPix=16)

if 0: #G214
	CO12FITS="/home/qzyan/WORK/myDownloads/testScimes/G214CO12.fits"
	dendroFITS="/home/qzyan/WORK/myDownloads/testScimes/G214CO12Dendro.fits"
	dendroCat= "/home/qzyan/WORK/myDownloads/testScimes/G214CO12dendroCat.fit"

if 1: #complicatedTest
	CO12FITS="/home/qzyan/WORK/myDownloads/testScimes/complicatedTest/complicatedTestmasked.fits"
	dendroFITS="/home/qzyan/WORK/myDownloads/testScimes/complicatedTest/dendroSave500.fits"
	dendroCat= "/home/qzyan/WORK/myDownloads/testScimes/complicatedTest/dendroSave500.fit"





d=Dendrogram.load_from( dendroFITS )
criteriaVe=[doCloud.myVrms]
scaleVe=[25]

doCloud.doSCIMES(CO12FITS,dendroFITS,dendroCat,"ComplicateVoLu",  inputD=d)
doCloud.doSCIMES(CO12FITS,dendroFITS,dendroCat,"ComplicateVe", criteriaUsed=criteriaVe,scales=scaleVe,inputD=d)