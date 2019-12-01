
import os
import numpy as np
from astropy.table import Table,vstack
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import measure
#fast way to perform DBSCAN
#
from progressbar import *
import math
from myPYTHON import *
from skimage.morphology import watershed
import sys
from skimage.morphology import erosion, dilation
from scipy.ndimage import label, generate_binary_structure,binary_erosion,binary_dilation
from sklearn.cluster import DBSCAN
from madda import  myG210
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText



doG210 = myG210()

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))



class myDBSCAN(object):

	rms = 0.5
	TBModel="minV3minP16_dendroCatTrunk.fit"

	def __index__(self):
		pass

	def sumEdgeByCon1(self,extendMask): #7 in total
		raw=extendMask[1:-1,1:-1,1:-1]

		leftShiftZ=extendMask[0:-2, 1:-1, 1:-1]
		rightShiftZ=extendMask[2:, 1:-1 ,1:-1]

		leftShiftY=extendMask[1:-1, 0 : -2, 1:-1]
		rightShiftY=extendMask[1:-1, 2 : ,1:-1]

		leftShiftX=extendMask[1:-1, 1:-1,  0:-2]
		rightShiftX=extendMask[1:-1, 1:-1 , 2: ]


		sumAll=raw+leftShiftZ+rightShiftZ+leftShiftY+rightShiftY+leftShiftX+rightShiftX

		return  sumAll



	def sumEdgeByCon2(self,extendMask): #27 in total
		sumAll=extendMask[1:-1,1:-1,1:-1]*0
		Nz,Ny,Nx= sumAll.shape
		for i in [-1,0,1]:
			for j in [-1,0,1]:
				for k in [-1,0,1]:

					if np.sqrt( abs(i)+abs(j)+abs(k))>1.5:
						continue

					sumAll=sumAll+  extendMask[ 1+i:Nz+1+i , j+1:Ny+1+j , k+1: Nx+1+k  ]

		return  sumAll



	def sumEdgeByCon3(self,extendMask): #27 in total
		raw=extendMask[1:-1,1:-1,1:-1]
		Nz,Ny,Nx= raw.shape
		sumAll=raw*0
		for i in [-1,0,1]:
			for j in [-1,0,1]:
				for k in [-1,0,1]:
					sumAll=sumAll+  extendMask[ 1+i:Nz+1+i , j+1:Ny+1+j , k+1: Nx+1+k  ]


		return  sumAll




	def slowDBSCAN(self,COdata,COHead, min_sigma=2, min_pix=16, connectivity=2 ,region="" ,saveFITS=None ):
		"""
		Use the sklearn DBSCAN to calculate, just for comparison, to test the computeDBSCAN is right
		:param COdata:
		:param COHead:
		:param min_sigma:
		:param min_pix:
		:param connectivity:
		:param region:
		:return:
		"""
		###

		goodIndices=np.where(COdata>= min_sigma*self.rms   )

		coordinates= zip( goodIndices[0] , goodIndices[1] ,goodIndices[2]  )

		#eps=1.5 form connectivity 2,
		if connectivity==2:
			db = DBSCAN(eps=1.5  , min_samples= min_pix      ).fit(coordinates)

		if connectivity==1:
			db = DBSCAN(eps=1.1  , min_samples= min_pix      ).fit(coordinates)
		if connectivity==3:
			db = DBSCAN(eps=1.8  , min_samples= min_pix      ).fit(coordinates)

		labels = db.labels_
		print min(labels),"minimumLabel?"
		#u,c= np.unique(labels,return_counts=True)

		#print len(u)

		mask=np.zeros_like(COdata)-1

		mask[goodIndices]= labels
		if saveFITS==None:
			fits.writeto("dbscanMask1Sigma.fits",mask,header=COHead,overwrite=True)

		else:
			fits.writeto(saveFITS ,mask,header=COHead,overwrite=True)




	def computeDBSCAN(self,COdata,COHead, min_sigma=2, min_pix=16, connectivity=2 ,region="" , getMask=False ):
		"""
		min_pix the the minimum adjacen number for are core point
		:param COdata:
		:param min_sigma:
		:param min_pix:
		:param connectivity:
		:return:
		"""
		#pass


		minValue = min_sigma*self.rms



		Nz,Ny,Nx  = COdata.shape
		extendMask = np.zeros([Nz+2,Ny+2,Nx+2] ,dtype=int)

		extendMask[1:-1,1:-1,1:-1] = COdata>=minValue    #[COdata>=minValue]=1

		s=generate_binary_structure(3,connectivity)


		if connectivity==1:
			coreArray=self.sumEdgeByCon1(extendMask)

		if connectivity==2:
			coreArray=self.sumEdgeByCon2(extendMask)

		if connectivity==3:
			coreArray=self.sumEdgeByCon3(extendMask)

		coreArray = coreArray>=min_pix
		coreArray[ COdata<minValue  ]=False #remove falsely, there is a possibility that, a bad value may have lots of pixels around
		coreArray=coreArray+0

		labeled_core, num_features=label(coreArray,structure=s) #first label core, then expand, otherwise, the expanding would wrongly connected
		selectExpand= np.logical_and(labeled_core==0, COdata>=minValue  )
		#expand labeled_core
		#coreLabelCopy=labeled_core.copy()

		expandTry = dilation(labeled_core , s  ) # first try to expand, then only keep those region that are not occupied previously

		labeled_core[  selectExpand  ] =  expandTry[ selectExpand  ]


		#allArray[COdata<minValue ]=False #remove falsely expanded values
		#allArray=labeled_core+0


		labeled_array = labeled_core
		saveName="{}dbscanS{}P{}Con{}.fits".format( region,min_sigma,min_pix,connectivity )




		if getMask:

			return labeled_array>0 #actually return mask


		print num_features,"features found!"

		fits.writeto(saveName, labeled_array, header=COHead, overwrite=True)
		return saveName



	def maskByGrow(self,COFITS,peakSigma=3,minV=1.):

		COData,COHead=myFITS.readFITS( COFITS )
		markers=np.zeros_like(COData )

		COData[COData<minV* self.rms]=0

		markers[COData>peakSigma*self.rms] = 1

		labels=watershed(COData,markers)
		fits.writeto("growMaskPeak3Min1.fits",labels,header=COHead,overwrite=True)


	def myDilation(self,scimesFITS,rawCOFITS,startSigma=20,endSigma=2,saveName=""):
		"""
		#because SCIMES removes weak emissions in the envelop of clouds, we need to add them back
		#one possible way is to use svm to split the trunk, test this con the  /home/qzyan/WORK/myDownloads/MWISPcloud/ClusterAsgn_ComplicateVe.fits

		:return:
		"""

		#cloudData,cloudHead = myFITS.readFITS("/home/qzyan/WORK/myDownloads/MWISPcloud/ClusterAsgn_ComplicateVe.fits")

		cloudData,cloudHead = myFITS.readFITS(scimesFITS)

		#rawFITS= rawCOFITS #"/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"

		rawCO,rawHead=   myFITS.readFITS( rawCOFITS )

		#the expansion should stars from high coValue, to low CO values, to avoid cloud cross wak bounarires
		#sCon=generate_binary_structure(3,2)
		print "Expanding clous..."
		for sigmas in np.arange(startSigma,endSigma-1,-1):

			#produceMask withDBSCAN
			if sigmas>2:
				COMask = self.computeDBSCAN( rawCO,rawHead, min_sigma=sigmas, min_pix=8, connectivity=2 ,region="" , getMask=True )

			else:
				COMask = self.computeDBSCAN(  rawCO,rawHead, min_sigma=sigmas, min_pix=16, connectivity=2 ,region="" , getMask=True )

			for i in range(2000):
				rawAssign=cloudData.copy()
				cloudData=cloudData+1 #to keep reagion that has no cloud as 0

				d1Try=dilation(cloudData  ) #expand with connectivity 1, connectivity 2, expandong two fast

				assignRegion= np.where(np.logical_and(cloudData==0 , COMask ) )

				cloudData[ assignRegion ] = d1Try[ assignRegion ]

				cloudData=cloudData-1

				diff= rawAssign-cloudData

				print  "Sigmas: {}, Loop:{}, difference:{}".format(sigmas,i,np.sum(diff))
				if np.sum(diff )==0:
					break


		fits.writeto( saveName+"_extend.fits",cloudData ,header=cloudHead,overwrite=True)





	def directLabel(self,COFITS,DBMaskFITS,min_sigma=3,min_pix=8,calCat=True ,useMask=True, peakSigma=3. ):

		saveMarker=""
		COData,COHead=myFITS.readFITS( CO12FITS )

		if useMask:
			DBMaskData,_=  myFITS.readFITS(  DBMaskFITS )

			maskData=np.zeros_like( DBMaskData )

			maskData[COData>min_sigma*self.rms]=1
			maskData[DBMaskData==0]=0
			saveLabel= "LabelSigma_{}_P{}.fits".format( min_sigma,min_pix )

		else:

			#use peak sigma to grow a mask

			maskData=np.zeros_like( COData )
			maskData[COData>min_sigma*self.rms]=1


			saveLabel= "NoMaskLabelSigma_{}_P{}.fits".format( min_sigma,min_pix )

			saveMarker="growMask"

		labels=measure.label(maskData,connectivity=1)
		fits.writeto(  saveLabel, labels,header=COHead, overwrite=True)


		if calCat  :

			self.getCatFromLabelArray(COFITS,saveLabel,self.TBModel,saveMarker=saveMarker,  minPix=min_pix,rms= min_sigma  )


	def getCatFromLabelArray(self,  CO12FITS,labelFITS,TBModel,minPix=8,rms=2 ,saveMarker="", peakSigma=3. ):
		"""
		Extract catalog from
		:param labelArray:
		:param head:
		:return:
		"""



		if saveMarker=="":

			saveName= "Sigma{}_P{}FastDBSCAN.fit".format(rms,minPix)

		else:
			saveName=saveMarker+".fit"

		clusterTBOld=Table.read( TBModel )

		###
		dataCO, headCO = myFITS.readFITS( CO12FITS )

		dataCO=np.nan_to_num(dataCO)


		dataCluster , headCluster=myFITS.readFITS( labelFITS )


		wcsCloud=WCS( headCluster )

		clusterIndex1D= np.where( dataCluster>0)
		clusterValue1D=  dataCluster[clusterIndex1D ]

		Z0,Y0,X0 = clusterIndex1D

		newTB= Table( clusterTBOld[0])
		newTB["sum"]=newTB["flux"]

		newTB["l_rms"]=newTB["v_rms"]
		newTB["b_rms"]=newTB["v_rms"]

		newTB["pixN"]=newTB["v_rms"]
		newTB["peak"]=newTB["v_rms"]

		dataClusterNew=np.zeros_like( dataCluster)

		# in the newCluster, number stars from 1, not zero

		idCol="_idx"


		#count all clusters

		#ids,count=np.unique(dataCluster,return_counts=True )
		ids,count=np.unique(clusterValue1D,return_counts=True )

		GoodIDs=  ids[count>=minPix ]

		GoodCount = count[ count>=minPix  ]



		print "Total number of turnks? ",len(ids)
		print "Total number of Good Trunks? ",len(GoodIDs)

		#dataCO,headCO=doFITS.readFITS( CO12FITS )
		widgets = ['Recalculating cloud parameters: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),  ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

		pbar = ProgressBar(widgets=widgets, maxval=len(GoodIDs))
		pbar.start()


		catTB=newTB.copy()

		catTB.remove_row(0)

		runIndex=0

		for i in  range(len(GoodIDs)) :

			#i would be the newID
			newID= GoodIDs[i]

			if newID==0:
				continue

			pixN=GoodCount[i]

			newRow=newTB[0]


			newRow[idCol] = newID

			cloudIndex=self.getIndices(Z0,Y0,X0,clusterValue1D,newID)

			coValues=  dataCO[ cloudIndex ]

			peak=np.max( coValues)

			#if peak<minPeak:

				#pbar.update(runIndex)
				#runIndex=runIndex+1

				#continue


			cloudV=cloudIndex[0]
			cloudB=cloudIndex[1]
			cloudL=cloudIndex[2]


			sumCO=np.sum( coValues )

			Vcen,Vrms= weighted_avg_and_std(cloudV, coValues )
			Bcen,Brms= weighted_avg_and_std(cloudB, coValues )
			Lcen,Lrms= weighted_avg_and_std(cloudL, coValues )

			#calculate the exact area

			LBcore=zip(  cloudB ,    cloudL   )

			pixelsN= {}.fromkeys(LBcore).keys() #len( set(LBcore) )
			area_exact=len(pixelsN)*0.25 #arc mins square


			dataClusterNew[cloudIndex] =newID

			#save values
			newRow["pixN"]= pixN
			newRow["peak"]= peak

			newRow["sum"]= sumCO
			newRow["area_exact"]= area_exact

			newRow["x_cen"],  newRow["y_cen"], newRow["v_cen"]= wcsCloud.wcs_pix2world( Lcen, Bcen,Vcen ,0)
			newRow["v_cen"]= newRow["v_cen"]/1000.
			dv=headCluster["CDELT3"]/1000. #km/s

			dl= abs( headCluster["CDELT1"] ) #deg

			newRow["v_rms"] = Vrms*dv

			newRow["l_rms"] = Lrms*dl
			newRow["b_rms"] = Brms*dl

			catTB.add_row(newRow)

			pbar.update(runIndex)
			runIndex=runIndex+1


		pbar.finish()
		#save the clouds

		#fits.writeto(self.regionName+"NewCloud.fits", dataClusterNew,header=headCluster,overwrite=True   )
		catTB.write( saveName ,overwrite=True)


	def getIndices(self,Z0,Y0,X0,values1D,choseID):



		cloudIndices = np.where(values1D==choseID )

		cX0=X0[cloudIndices ]
		cY0=Y0[cloudIndices ]
		cZ0=Z0[cloudIndices ]

		return tuple( [ cZ0, cY0, cX0 ]  )





	def draw(self ):
		"""
		#draw compare of
		:return:
		"""



		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		drawTB=Table.read( "Sigma1_P25FastDendro.fit" )


		axNumber=fig.add_subplot(1,2,1)




		axArea= fig.add_subplot(1,2,2)

		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )


		totalTB=  [drawTB] #TBList1+TBList2

		for i in range( len(totalTB) ):

			eachTB = totalTB[i]

			binN,binEdges=np.histogram(eachTB["area_exact"]/3600., bins=areaEdges  )


			axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8  ,alpha= 0.5 )


		axArea.set_yscale('log')
		axArea.set_xscale('log')


		axArea.legend()



		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Bin number of trunks ")


		plt.savefig( "compareDendroParaDBMask.pdf" ,  bbox_inches='tight')
		plt.savefig( "compareDendroParaDBMask.png" ,  bbox_inches='tight',dpi=300)


	def getEdgeCenter(self,edges):

		areaCenters= ( edges[1:] + edges[0:-1] )/2.

		return  areaCenters

	def drawDBSCANArea(self):

		TB2_16= "G2650CO12DBCatS2P16Con2.fit"
		#TB2_16= "DBSCAN2_9Sigma1_P1FastDBSCAN.fit"
		TB25_9="G2650CO12DBCatS2.5P9Con2.fit"
		TB35_9="G2650CO12DBCatS3.5P9Con2.fit"
		TB45_9="G2650CO12DBCatS4.5P9Con2.fit"
		TB55_9="G2650CO12DBCatS5.5P9Con2.fit"
		TB65_9="G2650CO12DBCatS6.5P9Con2.fit"
		TB75_9="G2650CO12DBCatS7.5P9Con2.fit"

		TB3_9= "G2650CO12DBCatS3.0P9Con2.fit"
		TB4_9= "G2650CO12DBCatS4.0P9Con2.fit"
		TB5_9= "G2650CO12DBCatS5.0P9Con2.fit"
		TB6_9= "G2650CO12DBCatS6.0P9Con2.fit"
		TB7_9= "G2650CO12DBCatS7.0P9Con2.fit"



		TBFiles=[TB2_16,TB25_9,TB3_9, TB35_9, TB4_9, TB45_9,TB5_9, TB55_9, TB6_9 , TB65_9, TB7_9, TB75_9   ]



		sigmas=[2,2.5,3,3.5,4,4.5, 5, 5.5, 6, 6.5,7,7.5]

		labelStr=[  r"2$\sigma$, P16" ,   r"2.5$\sigma$, P16" ,  r"3$\sigma$, P16" ,  r"3.5$\sigma$, P16" ,   r"4$\sigma$, P16" ,  r"4.5$\sigma$, P16" , \
		            r"5$\sigma$, P16" ,  r"5.5$\sigma$, P16" , r"6$\sigma$, P16"  , r"6.5$\sigma$, P16"  , r"7$\sigma$, P16", r"7.5 $\sigma$, P16"   ]


		TBList=[]



		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )
		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })



		axArea=fig.add_subplot(1,2,1)



		for eachTBF,eachLab in zip(TBFiles,labelStr):
			tb=Table.read(eachTBF)

			tb=self.removeWrongEdges(tb)

			TBList.append( tb )


			goodT=tb


			goodT=goodT[ goodT["pixN"]>=16 ]

			goodT=goodT[ goodT["peak"]>=1.5 ]

			#
			binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )




			axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,label=eachLab ,alpha= 0.5 )


		axArea.set_yscale('log')
		axArea.set_xscale('log')

		axArea.legend()
		axArea.set_title("Plot of Area distribution with DBSCAN")

		plt.savefig( "dbscanArea.png" ,  bbox_inches='tight',dpi=300)


	def drawDBSCANNumber(self):

		minPix=8

		TB2_16="G2650CO12DBCatS2.0P{}Con2.fit".format(minPix)
		TB25_9="G2650CO12DBCatS2.5P{}Con2.fit".format(minPix)
		TB35_9="G2650CO12DBCatS3.5P{}Con2.fit".format(minPix)
		TB45_9="G2650CO12DBCatS4.5P{}Con2.fit".format(minPix)
		TB55_9="G2650CO12DBCatS5.5P{}Con2.fit".format(minPix)
		TB65_9="G2650CO12DBCatS6.5P{}Con2.fit".format(minPix)
		TB75_9="G2650CO12DBCatS7.5P{}Con2.fit".format(minPix)

		TB3_9= "G2650CO12DBCatS3.0P{}Con2.fit".format(minPix)
		TB4_9= "G2650CO12DBCatS4.0P{}Con2.fit".format(minPix)
		TB5_9= "G2650CO12DBCatS5.0P{}Con2.fit".format(minPix)
		TB6_9= "G2650CO12DBCatS6.0P{}Con2.fit".format(minPix)
		TB7_9= "G2650CO12DBCatS7.0P{}Con2.fit".format(minPix)


		TBFiles=[TB2_16,TB25_9,TB3_9, TB35_9, TB4_9, TB45_9,TB5_9, TB55_9, TB6_9 , TB65_9, TB7_9, TB75_9   ]
		TBList=[]

		Nlist=[]

		sigmas=[2,2.5,3,3.5,4,4.5, 5, 5.5, 6, 6.5,7,7.5]

		for eachTBF in TBFiles:
			tb=Table.read(eachTBF)
			TBList.append( tb )
			goodT=tb
			goodT=goodT[ goodT["pixN"]>=16 ]

			goodT=goodT[ goodT["peak"]>=1.5 ]

			Nlist.append(len(goodT) )

		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		drawTB=Table.read( "Sigma1_P25FastDendro.fit" )


		axNumber=fig.add_subplot(1,2,1)

		axNumber.plot(sigmas,Nlist,'o-',color='blue')

		axNumber.set_ylabel(r"Total number of trunks")
		axNumber.set_xlabel(r"CO cutoff ($\sigma$)")
		axNumber.set_title("Plot of total trunk numbers with DBSCAN")

		plt.savefig( "dbscanNumber.png" ,  bbox_inches='tight',dpi=300)



	def drawAreaDistribute(self,TBName,region="",algorithm='Dendrogram'):
		"""

		:return:
		"""

		TB=Table.read( TBName )

		TBLOcal=Table.read("DBSCAN35_9Sigma1_P1FastDBSCAN.fit")
		TBAll=vstack([TB,TBLOcal ])

		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )



		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })
		axArea=fig.add_subplot(1,1,1)

		##########
		goodT=TB

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5, label=region  )


		self.getAlphaWithMCMC( goodT["area_exact"] )

		###############

 		goodT=TBAll

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5, label="All"  )



		###############
 		goodT=TBLOcal

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Velocity range (0-30 km/s)"  )










		###############

		axArea.set_yscale('log')
		axArea.set_xscale('log')
		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Bin number of trunks ")



		axArea.legend()
		axArea.set_title("Plot of Area distribution with DBSCAN")

		plt.savefig( region+"dbscanArea.png" ,  bbox_inches='tight',dpi=300)


	def getAlphaWithMCMC(self,areaArray,minArea=0.03,maxArea=1. ):
		"""
		areaArray should be in square armin**2
		:param areaArray:
		:param minArea:
		:param maxArea:
		:return:
		"""

		print "Fitting index with MCMC..."
		areaArray=areaArray/3600.
		select=np.logical_and( areaArray>minArea, areaArray<maxArea)
		rawArea =   areaArray[ select ]

		doG210.fitPowerLawWithMCMCcomponent1(rawArea,minV=minArea,maxV=maxArea)



	def drawSumDistribute(self,TBName,region=""):
		"""
		:return:
		"""

		TB=Table.read( TBName )

		TBLOcal=Table.read("DBSCAN35_9Sigma1_P1FastDBSCAN.fit")
		TBAll=vstack([TB,TBLOcal ])



		goodT=TB





		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })
		axArea=fig.add_subplot(1,1,1)

		##########
		pixNCol =goodT["flux"]

		logPixN=np.log10( pixNCol  )

		print min( logPixN  ),max( logPixN  )

		areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,100)
		areaCenter=self.getEdgeCenter( areaEdges )

		binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


		drawBind=binN[binN>0]
		drawCenter= areaCenter[binN>0]

		axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Flux"  )

		select=np.logical_and( drawCenter<6, drawCenter>4 )
		x=drawCenter[ select ]   #np.log(drawCenter)
		y=np.log10(drawBind  )[ select]

		#print np.polyfit(x,y,1)

		###########################



		if 0:
			tbVox=goodT[ goodT["pixN"]>16  ]
			pixNCol =tbVox["pixN"]

			logPixN=np.log10( pixNCol  )

			print min( logPixN  ),max( logPixN  )

			areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,100)
			areaCenter=self.getEdgeCenter( areaEdges )

			binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


			drawBind=binN[binN>0]
			drawCenter= areaCenter[binN>0]

			axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Voxel"  )

			select=np.logical_and( drawCenter<4, drawCenter>1.5 )
			x=drawCenter[ select ]   #np.log(drawCenter)
			y=np.log10(drawBind  )[ select]

			print np.polyfit(x,y,1)

		###########################




		##########
		tbVox=goodT
		pixNCol =tbVox["area_exact"]

		logPixN=np.log10( pixNCol  )
		print "draw areas"
		print min( logPixN  ),max( logPixN  )

		areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,15)

		print  areaEdges

		areaCenter=self.getEdgeCenter( areaEdges )

		binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


		drawBind=binN[binN>0]
		drawCenter= areaCenter[binN>0]

		axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="area exact aa "  )

		select=np.logical_and( drawCenter<4, drawCenter>1.5  )


		x=drawCenter[ select ]   #np.log(drawCenter)
		y=np.log10(drawBind  )[ select]

		a= np.polyfit(x,y,1)
		p=np.poly1d(a)
		axArea.plot(  drawCenter ,  p(drawCenter) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5   )

		print a
		###########################








		axArea.set_xlabel(r"Voxel Number")
		axArea.set_ylabel(r"Bin number of trunks ")



		axArea.legend()
		axArea.set_title("Plot of Pixel distribution with DBSCAN")

		plt.savefig( region+"dbscanTotalPixel.png" ,  bbox_inches='tight',dpi=300)





	def drawPixNDistribute(self,TBName,region=""):
		"""

		:return:
		"""

		TB=Table.read( TBName )

		TBLOcal=Table.read("DBSCAN35_9Sigma1_P1FastDBSCAN.fit")
		TBAll=vstack([TB,TBLOcal ])



		goodT=TB
		goodT=goodT[ goodT["pixN"]>=16 ]

		pixNCol =goodT["pixN"]

		logPixN=np.log10( pixNCol  )

		print logPixN



		areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,100)
		areaCenter=self.getEdgeCenter( areaEdges )



		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })
		axArea=fig.add_subplot(1,1,1)

		##########

		binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


		drawBind=binN[binN>0]
		drawCenter= areaCenter[binN>0]

		axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label=" ??"  )

		select=np.logical_and( drawCenter<4, drawCenter>1.5 )
		x=drawCenter[ select ]   #np.log(drawCenter)
		y=np.log(drawBind  )[ select]

		print np.polyfit(x,y,1)

		###############


		axArea.set_xlabel(r"Voxel Number")
		axArea.set_ylabel(r"Bin number of trunks ")



		axArea.legend()
		axArea.set_title("Plot of Pixel distribution with DBSCAN")

		plt.savefig( region+"dbscanTotalPixel.png" ,  bbox_inches='tight',dpi=300)





	def roughFit(self,centers,bins ):
		"""

		:return:
		"""
		y= bins[bins>0 ]  # areaCenter[binN>0]
		x=  centers[ bins>0  ]




		x1= x[x<= 0.1]  # areaCenter[binN>0]
		y1=  y[x<= 0.1 ]


		x2= x1[x1>=0.005 ]  # areaCenter[binN>0]
		y2=  y1[x1>= 0.005 ]

		x=np.log10(x2)
		y=np.log10(y2)


		print x
		print y

		print np.polyfit(x,y,1)


		return




	def drawTrueArea(self):

		goodTB=Table.read( "/home/qzyan/WORK/projects/maddalena/dendroDisPath/G2650/G2650goodDisTB.fit"  )
		dendroTB=Table.read( "/home/qzyan/WORK/myDownloads/testScimes/mosaicV1NewTB.fit" )

		areas=[]

		for eachG in goodTB:

			d=eachG["distance"]
			ID=int( eachG["sourceName"].split('oud')[1]  )
			dendroRow=  dendroTB[ID-1 ]

			area=dendroRow["area_exact"]

			area/0.25*(d*np.radians( 0.5/60. ) )**2
			#print area


			areas.append(area )
		#plot
		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		drawTB=Table.read( "Sigma1_P25FastDendro.fit" )

		bins=np.linspace( np.min(areas),np.max(areas),5  )
		areaCenter=self.getEdgeCenter( bins )

		ax=fig.add_subplot(1,2,1)

		binN,binEdges=np.histogram(areas, bins=bins  )

		ax.scatter(areaCenter,binN   )

		ax2=fig.add_subplot(1,2,2)


		ax2.scatter( goodTB["vlsr"], goodTB["distance"]  )


		plt.savefig( "exactArea.png" ,  bbox_inches='tight',dpi=300)




	def removeWrongEdges(self,TB):



		processTB=TB.copy()

		#remove cloudsThat touches the noise edge of the fits


		#part1= processTB[ np.logical_and( processTB["x_cen"]>=2815 ,processTB["y_cen"]>= 1003  )   ] #1003, 3.25

		#part2= processTB[ np.logical_and( processTB["x_cen"]<= 55 ,processTB["y_cen"]>= 1063  )   ] #1003, 3.25

		if "peak" in TB.colnames: #for db scan table

			part1= processTB[ np.logical_or( processTB["x_cen"]>26.25 ,processTB["y_cen"] < 3.25  )   ] #1003, 3.25

			part2= part1[ np.logical_or( part1["x_cen"]<49.25 ,part1["y_cen"]<  3.75 )   ] #1003, 3.25

			return part2
		else: #dendrogram tb

			part1= processTB[ np.logical_or( processTB["x_cen"]< 2815 ,processTB["y_cen"] < 1003  )   ] #1003, 3.25

			part2= part1[ np.logical_or( part1["x_cen"]>  55 ,part1["y_cen"]< 1063  )   ] #1003, 3.25

			return part2

	def removeAllEdges(self,TBList):
		"""

		:param TBList:
		:return:
		"""
		newList=[]

		for eachTB in TBList:
			newList.append( self.removeWrongEdges(eachTB) )

			
		return newList

	def getNList(self,TBList):

		Nlist=[]

		for eachTB in TBList:
			Nlist.append( len(eachTB) )
		return Nlist


	def areaAndNumberDistribution(self, algorithm="Dendrogram" ):
		"""
		#draw the area the
		:return:
		"""


		#first, get TBList

		tb8,tb16,label8,label16,sigmaList=self.getTBList(algorithm=algorithm)

		tb8=self.removeAllEdges(tb8)
		tb16=self.removeAllEdges(tb16)


		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		axNumber=fig.add_subplot(1,2,1)

		Nlist8=self.getNList(tb8)
		Nlist16=self.getNList(tb16)


		axNumber.plot(sigmaList,Nlist8,'o-',label="MinPix = 8",color="blue",lw=0.5)
		axNumber.plot(sigmaList,Nlist16,'o-',label="MinPix = 16",color="green", lw=0.5)





		#axArea.set_xlabel(r"Area (deg$^2$)")
		axNumber.set_ylabel(r"Total number of trunks")
		axNumber.set_xlabel(r"CO cutoff ($\sigma$)")

		axNumber.legend()

		at = AnchoredText(algorithm, loc=3, frameon=False)
		#at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
		axNumber.add_artist(at)


		################ Area ##############


		axArea= fig.add_subplot(1,2,2)

		areaEdges=np.linspace(0,150,6000)
		areaCenter=self.getEdgeCenter( areaEdges )


		totalTB=tb8+tb16
		labelStr=label8+label16

		for i in range( len(totalTB) ):

			eachTB = totalTB[i]

			binN,binEdges=np.histogram(eachTB["area_exact"]/3600., bins=areaEdges  )


			axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,label=labelStr[i] ,alpha= 0.5 )


		#set ticikes of Area


		axArea.set_yscale('log')
		axArea.set_xscale('log')

		axArea.set_xlim( [ 0.01,150 ] )


		if algorithm=="DBSCAN":
			axArea.set_ylim( [ 0.8,50000 ] )

		else:
			axArea.set_ylim( [ 0.8,10000 ] )

		axArea.legend(ncol=2)



		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Bin number of trunks ")















		plt.savefig( "comparePara_{}.pdf".format(algorithm) ,  bbox_inches='tight')

		plt.savefig( "comparePara_{}.png".format(algorithm) ,  bbox_inches='tight',dpi=300)



	def getTBList(self, algorithm="DBSCAN"):
		"""
		return a list of table,
		:param minP:
		:param algorithm:
		:return:
		"""

		if algorithm=="DBSCAN":
			#ignore minP, only has 8


			TBList=[]
			TBList16=[]

			TBLabelsP8=[]
			TBLabelsP16=[]
			minPix=8


			DbscanSigmaList= np.arange(2,6.5,0.5)

			for sigmas in DbscanSigmaList:
				tbName= "G2650CO12DBCatS{:.1f}P{}Con2.fit".format(sigmas, minPix)
				ttt8=Table.read(tbName)
				TBList.append(ttt8  )
				ttt16=ttt8[ttt8["pixN"]>=16]
				TBList16.append(ttt16  )
				TBLabelsP8.append(  r"{:.1f}$\sigma$, P8".format( sigmas)   )
				TBLabelsP16.append( r"{:.1f}$\sigma$, P16".format( sigmas)   )

			return TBList,TBList16,TBLabelsP8,TBLabelsP16,DbscanSigmaList


		else:

			TBListP8=[]
			TBListP16=[]

			TBLabelsP8=[]
			TBLabelsP16=[]

			dendroSigmaList=[2,2.5 , 3,4,5,6]


			for sigmas in dendroSigmaList:
				tbName8= "minV{}minP{}_dendroCatTrunk.fit".format(sigmas, 8)
				tbName16= "minV{}minP{}_dendroCatTrunk.fit".format(sigmas, 16)

				TBListP8.append(Table.read(tbName8)  )
				TBListP16.append(Table.read(tbName16)  )


				TBLabelsP8.append(  r"{:.1f}$\sigma$, P8".format( sigmas)   )
				TBLabelsP16.append( r"{:.1f}$\sigma$, P16".format( sigmas)   )



			return TBListP8,TBListP16,TBLabelsP8,TBLabelsP16,dendroSigmaList





	def ZZ(self):
		pass


doDBSCAN=myDBSCAN()

G2650CO12FITS="/home/qzyan/WORK/myDownloads/testFellwalker/WMSIPDBSCAN/G2650Local30.fits"
DBMaskFITS= "/home/qzyan/WORK/myDownloads/testFellwalker/G2650DB_1_25.fits"
TaurusCO12FITS="/home/qzyan/WORK/dataDisk/Taurus/t12_new.fits"
PerCO12="/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650Per3060.fits"

localCO13="/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650Local30CO13.fits"

G210CO12="/home/qzyan/WORK/myDownloads/newMadda/data/G210CO12sm.fits"
G210CO13="/home/qzyan/WORK/myDownloads/newMadda/data/G210CO13sm.fits"


if 1:
	doDBSCAN.areaAndNumberDistribution(algorithm="Dendrogram")
	doDBSCAN.areaAndNumberDistribution(algorithm="DBSCAN")


if 0: # get catalog from extended fits


	doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650DisCloudVe20_extend.fits",doDBSCAN.TBModel,saveMarker="G2650CloudForDisCat")
	sys.exit()



if 0:


	doDBSCAN.drawDBSCANNumber()
	doDBSCAN.drawDBSCANArea()


if  0: #dilation

	#scimesFITS= "/home/qzyan/WORK/myDownloads/MWISPcloud/ClusterAsgn_ComplicateVe.fits"
	#rawFITS="/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"

	scimesFITS= "/home/qzyan/WORK/myDownloads/MWISPcloud/distanceDendro/ClusterAsgn_Ve20.fits"
	rawFITS= G2650CO12FITS  #"/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"

	doDBSCAN.myDilation( scimesFITS , rawFITS, saveName="G2650DisCloudVe20",startSigma=10)

	sys.exit()

if 0:
	#doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650CO12dbscanS2P16Con2.fits",doDBSCAN.TBModel, saveMarker="G2650CO12DBCatS2P16Con2" )
	for i in np.arange(2 ,8,0.5):
		savename="G2650CO12DBCatS{}P{}Con2".format(i,8)
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650CO12dbscanS{}P8Con2.fits".format(i),doDBSCAN.TBModel,saveMarker=savename)





if 0:
	COData,COHead=myFITS.readFITS( G2650CO12FITS)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=2,min_pix=16,connectivity=2,region="G2650CO12")

	for i in np.arange(2 ,8,0.5):
		doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=i,min_pix=8,connectivity=2,region="G2650CO12")

	sys.exit()





if 0: #
	G214COFITS="G214CO12.fits"
	COData,COHead=myFITS.readFITS( G214COFITS)

	doDBSCAN.computeDBSCAN(COData, COHead,region="G214")
	doDBSCAN.slowDBSCAN(COData, COHead,region="G214")




	sys.exit()






if 0:# DBSCAN for G210
	region="G210CO13"

	processFITS=G210CO13

	doDBSCAN.rms=0.25


	if 1:#find clouds
		COData,COHead=myFITS.readFITS( processFITS)

		for sigmas in [3,4,5]:
			saveName=doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=sigmas, min_pix=8,connectivity=2, region=region)

			doDBSCAN.getCatFromLabelArray(processFITS, saveName , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN{}_8".format(sigmas)   )

	sys.exit()




if 0:# DBSCAN for CO13
	region="Local13"
	doDBSCAN.rms=0.25


	if 0:#find clouds
		COData,COHead=myFITS.readFITS( localCO13)

		for sigmas in [3,4,5]:
			doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=sigmas, min_pix=9,connectivity=2, region=region)

		sys.exit()

	else:#calcatelog

		doDBSCAN.getCatFromLabelArray(localCO13,  "Local13dbscanS3P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN3_9"  )
		doDBSCAN.getCatFromLabelArray(localCO13,  "Local13dbscanS4P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN4_9"  )
		doDBSCAN.getCatFromLabelArray(localCO13,  "Local13dbscanS5P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN5_9"  )



if 0:
	#draw perseus
	#doDBSCAN.drawAreaDistribute("DBSCAN3_9.fit"  )

	doDBSCAN.drawAreaDistribute("minV3minP16_dendroCatTrunk.fit" , region="Perseus" )



	#doDBSCAN.drawSumDistribute("DBSCAN3_9Sigma1_P1FastDBSCAN.fit"  )


	#doDBSCAN.drawSumDistribute("DBSCAN3_9.fit"  )



	#doDBSCAN.drawSumDistribute("minV3minP16_dendroCatTrunk.fit"  )
	#doDBSCAN.drawDBSCANArea()

	sys.exit()



if 0:# DBSCAN for perseus
	region="PerG2650"
	PerCO12="/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650Per3060.fits"

	if 0:#find clouds


		COData,COHead=myFITS.readFITS( PerCO12)
		doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2, region=region)
		doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2, region=region)

		sys.exit()
	else:#calcatelog

		doDBSCAN.getCatFromLabelArray(PerCO12,  "PerG2650dbscanS3P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker="DBSCAN3_9"  )
		doDBSCAN.getCatFromLabelArray(PerCO12,  "PerG2650dbscanS4P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker="DBSCAN4_9"  )
		doDBSCAN.getCatFromLabelArray(PerCO12,  "PerG2650dbscanS5P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker="DBSCAN5_9"  )

if 0:#Example
	doDBSCAN.rms=0.5
	COData,COHead=myFITS.readFITS( CO12FITS)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=i,min_pix=9,connectivity=2)


if 0: #Taurus
	COData,COHead=myFITS.readFITS( TaurusCO12FITS)

	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=1,min_pix=25,connectivity=3)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=2,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=3,min_pix=9,connectivity=2)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=6,min_pix=9,connectivity=2)



if 0:
	ModelTB="minV3minP16_dendroCatTrunk.fit"

	#doDBSCAN.getCatFromLabelArray(CO12FITS, "dbscanS1P25Con3.fits",  ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN1_25"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS, "dbscanS2P16Con2.fits",  ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN2_16"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS3P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN3_9"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS4P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN4_9"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN5_9"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS6P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN6_9"  )

	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS2.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN25_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS3.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN35_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS4.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN45_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS5.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN55_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS6.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN65_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS7.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN75_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS7P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN7_9"  )

	import sys
	sys.exit()

if 0:
	COData,COHead=myFITS.readFITS( CO12FITS)

	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=1,min_pix=25,connectivity=3)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=2,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=3,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=6,min_pix=9,connectivity=2)







if 0:

	ModelTB="minV3minP16_dendroCatTrunk.fit"

	doDBSCAN.getCatFromLabelArray(CO12FITS,DBMaskFITS,  ModelTB,  rms=1,minPix=25 )