
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

		minValue=np.min(dataCluster )

		clusterIndex1D= np.where( dataCluster>minValue )
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

	def getSumToFluxFactor(self):

		theta =  np.deg2rad(0.5/60)
		omega = theta * theta
		f=115.271202000
		waveLength =299792458/(f*1e9)
		k= 1.38064852e3 #has converted to jansky
		factorSumToFlux=  2*k*omega/waveLength/waveLength
		
		return factorSumToFlux
	def converSumToFlux(self,sumRow):

		factorSumToFlux=self.getSumToFluxFactor(factorSumToFlux)

		return sumRow* factorSumToFlux #jansky

	def converFluxToSum(self, fluxRow):
		factorSumToFlux=self.getSumToFluxFactor(factorSumToFlux)

		return fluxRow/factorSumToFlux


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

	def cleanDBTB(self,dbTB,pixN=8,minV=3,minDelta=3):

		"""
		The minimum Peak, should be relative to the minValue
		:param dbTB:
		:param pixN:
		:param peak:
		:return:
		"""
		peakV=(minV + minDelta)*self.rms



		if type(dbTB)==list:
			newList=[]
			for eachT in dbTB:

				goodT=eachT.copy()
				goodT=goodT[ goodT["pixN"] >= pixN ]
				goodT=goodT[ goodT["peak"] >= peakV ]

				newList.append(goodT)

			return newList



		else:

			goodT=dbTB.copy()
			goodT=goodT[ goodT["pixN"] >= pixN ]
			goodT=goodT[ goodT["peak"] >= peakV ]

			return goodT





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

	def getRealArea(self,TB):
		print TB.colnames
		araeList=[]

		#print TB.colnames
		for eachR in TB:
			v = eachR["v_cen"]
			dis= (v- 4.24787753)/13.46359868*1000 # pc

			if dis<0  or dis> 1500 :
				continue

			N=eachR["area_exact"]/0.25


			#print N,eachR["pixN"]
			length= dis*np.deg2rad(0.5/60) # 0.0001454441043328608
			trueArea=length**2*N #eachR["pixN"]  #*10000
			#print N,  trueArea

			araeList.append(  trueArea )



		return np.asarray(araeList)



	def drawAreaDistribute(self,TBName,region="",algorithm='Dendrogram'):
		"""

		:return:
		"""

		TB=Table.read( TBName )

		TBLOcal=Table.read("DBSCAN35_9Sigma1_P1FastDBSCAN.fit")
		TBAll=vstack([TB,TBLOcal ])

		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )



		fig=plt.figure(figsize=(8,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })
		axArea=fig.add_subplot(1,1,1)

		##########
		goodT=TB

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>= self.rms*5. ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5, label= region)#   r"SCIMES,min3$\sigma$P16 12CO"  )
		print region
		self.getAlphaWithMCMC(  goodT["area_exact"] , minArea= 0.00065, maxArea=None , physicalArea=False )

		#a=np.linspace(1,3000,6000)

		#trueArea=1./a**2
		#self.getAlphaWithMCMC(  trueArea , minArea= 1e-7, maxArea=None , physicalArea=True )

		#print "Above???-2??"


		###############
 		goodT=TBLOcal

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Velocity range (0-30 km/s)12CO Raw"  )


		areaEdges=np.linspace(0,100,1000)
		areaCenter=self.getEdgeCenter( areaEdges )



		realArea=self.getRealArea(goodT)
		binN,binEdges=np.histogram( realArea  , bins=areaEdges  )

		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Velocity range (0-30 km/s)12CO, distance Corrected"  )
		print min(realArea),"The minimum area?"
		self.getAlphaWithMCMC(  realArea  ,minArea= 0.42836824657505895 , maxArea=None,  physicalArea=True)

		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )


		############### Perseus
 		goodT=  Table.read("Local13DBSCAN3_9.fit")

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label=r"(26$^\circ$-50$^\circ$)13CO"  )


		############### Perseus
 		goodT=  Table.read("G210DBSCAN3_9.fit")

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label=r"G210(210$^\circ$-220$^\circ$)12CO"  )

		###############
 		goodT=  Table.read("DBSCAN3_9.fit")

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Velocity range (30-60 km/s)12CO"  )









		###############

		axArea.set_yscale('log')
		axArea.set_xscale('log')
		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Bin number of trunks ")



		axArea.legend()
		axArea.set_title("Plot of Area distribution with DBSCAN")

		plt.savefig( region+"dbscanArea.png" ,  bbox_inches='tight',dpi=300)
		plt.savefig( region+"dbscanArea.pdf" ,  bbox_inches='tight' )


	def getAlphaWithMCMC(self,areaArray,minArea=0.03,maxArea=1.,physicalArea=False,verbose=True,plotTest=False,saveMark="" ):
		"""
		areaArray should be in square armin**2
		:param areaArray:
		:param minArea:
		:param maxArea:
		:return:
		"""

		print "Fitting index with MCMC..."

		if not physicalArea:
			areaArray=areaArray/3600.

		if maxArea!=None:
			select=np.logical_and( areaArray>minArea, areaArray<maxArea)

		else:
			select= areaArray>minArea

		rawArea =   areaArray[ select ]

		if verbose:
			print "Run first chain for {} molecular clouds.".format( len( rawArea ) )
		part1=doG210.fitPowerLawWithMCMCcomponent1(rawArea, minV=minArea, maxV=maxArea)
		if verbose:
			print "Run second chain for {} molecular clouds.".format( len(rawArea) )

		part2=doG210.fitPowerLawWithMCMCcomponent1(rawArea, minV=minArea, maxV=maxArea)

		allSample=np.concatenate(  [ part1 , part2 ]    )


		#test plot
		if plotTest:
			fig = plt.figure(figsize=(12, 6))
			ax0 = fig.add_subplot(1, 1, 1)
			# fig, axs = plt.subplots(nrows=1, ncols=2,  figsize=(12,6),sharex=True)
			rc('text', usetex=True)
			rc('font', **{'family': 'sans-serif', 'size': 13, 'serif': ['Helvetica']})

			ax0.scatter(part1,part2,s=10 )

			plt.savefig("mcmcSampleTest.pdf"  , bbox_inches='tight')
			aaaaaa

		meanAlpha= np.mean( allSample)
		stdAlpha=  np.std(allSample,ddof=1)
		if verbose:
			print "Alpha Mean: {:.2f}; std: {:.2f}".format( meanAlpha,  stdAlpha)

		return round(meanAlpha,2) , round(stdAlpha,2)

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



	def getTotalFluxList(self,TBList):

		fluxlist=[]

		for eachTB in TBList:

			if "sum" in eachTB.colnames:
				toalFlux=np.nansum( eachTB["sum"]  )*0.2 # K km/s

			else:
				toalFlux=np.nansum( eachTB["flux"]  )*0.2/self.getSumToFluxFactor() # K km/s

			fluxlist.append(  toalFlux  )
		return fluxlist




	def alphaDistribution(self):
		"""

		# draw alph distribution, for each DBSCAN,

		:return:
		"""

		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)



		fig=plt.figure(figsize=(15,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#############   plot dendrogram
		axDendro=fig.add_subplot(1,3,1)
		alphaDendro, alphaDendroError = self.drawAlpha( axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)

		at = AnchoredText(algDendro, loc=1, frameon=False)
		axDendro.add_artist(at)
		axDendro.set_ylabel(r"$\alpha$")
		axDendro.set_xlabel(r"CO cutoff ($\sigma$)")




		##############   plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro)
		#self.drawNumber(axDB,tb8DB,tb16DB, label8DB,  label16DB ,sigmaListDB)

		alphaDB,  alphaDBError = self.drawAlpha( axDB,tb8DB,tb16DB, label8DB,  label16DB ,sigmaListDB)

		at = AnchoredText(algDB, loc=1, frameon=False)
		axDB.add_artist(at)

		#axDB.set_ylabel(r"Total number of clusters")

		axDB.set_xlabel(r"CO cutoff ($\sigma$)")



		##########plot SCIMES

		allAlpha = alphaDB+alphaDendro
		allAlphaError = alphaDBError+alphaDendroError

		print "Average error, ",  np.mean(allAlphaError )

		errorAlpha= np.mean(allAlphaError )**2+  np.std(allAlpha,ddof=1)**2
		errorAlpha=np.sqrt( errorAlpha )

		alphaMean= np.mean( allAlpha)

		print "The mean alpha is {:.2f}, error is {:.2f}".format(alphaMean,errorAlpha )


		#draw Average Alpha

		axDendro.plot([min(sigmaListDB),max(sigmaListDB)],  [alphaMean, alphaMean],'g--',lw=1 )

		axDB.plot([min(sigmaListDB),max(sigmaListDB)],  [alphaMean, alphaMean],'g--',lw=1 )




		axDendro.legend(loc=3)
		axDB.legend(loc=3)

		fig.tight_layout()
		plt.savefig( "compareParaAlpha.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaAlpha.png"  ,  bbox_inches='tight',dpi=300)






	def fluxDistribution(self):
		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)

		compoleteFluxa= 9*(1500./250)**2*0.2*1.5 # 2 sigma


		#aaaaaa

		fig=plt.figure(figsize=(20,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,3,1)



		at = AnchoredText(algDendro, loc=3, frameon=False)
		axDendro.add_artist(at)

		self.drawFlux(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)

		axDendro.set_xlabel(r"Flux (K km s$^{-1}$)")
		axDendro.set_ylabel(r"Bin number of trunks ")


		axDendro.set_yscale('log')
		axDendro.set_xscale('log')
		#axDendro.plot( [compoleteFluxa,compoleteFluxa],[2,800],'--',color='black', lw=1  )



		axDendro.legend(loc=1, ncol=2 )
		#plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro )

		self.drawFlux(axDB,tb8DB,tb16DB, label8DB,label16DB, sigmaListDB)


		at = AnchoredText(algDB, loc=3, frameon=False)
		axDB.add_artist(at)

		axDB.set_xlabel(r"Flux (K km s$^{-1}$)")
		axDB.set_ylabel(r"Bin number of trunks ")

		axDB.set_yscale('log')
		axDB.set_xscale('log')

		axDB.legend(loc=1, ncol=2 )



		fig.tight_layout()
		plt.savefig( "compareParaFlux.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaFlux.png"  ,  bbox_inches='tight',dpi=300)







	def areaDistribution(self):

		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)



		#aaaaaa

		fig=plt.figure(figsize=(15,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,3,1)

		#self.drawNumber(axDendro,tb8Den,tb16Den,sigmaListDen)
		at = AnchoredText(algDendro, loc=3, frameon=False)
		axDendro.add_artist(at)

		self.drawArea(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)

		axDendro.set_xlabel(r"Area (deg$^2$)")
		axDendro.set_ylabel(r"Bin number of trunks ")


		axDendro.set_yscale('log')
		axDendro.set_xscale('log')

		compoleteArea= 9*(1500./250)**2*0.25/3600. #0.0225




		axDendro.plot( [compoleteArea,compoleteArea],[2,2000],'--',color='black', lw=1  )


		axDendro.legend(loc=1, ncol=2 )
		#plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro )
		#self.drawNumber(axDB,tb8DB,tb16DB,sigmaListDB)
		self.drawArea(axDB,tb8DB,tb16DB, label8DB,label16DB, sigmaListDB)


		at = AnchoredText(algDB, loc=3, frameon=False)
		axDB.add_artist(at)
		axDB.plot( [compoleteArea,compoleteArea],[2,2000],'--',color='black', lw=1  )
		axDB.set_xlabel(r"Area (deg$^2$)")
		axDB.set_ylabel(r"Bin number of trunks ")

		axDB.set_yscale('log')
		axDB.set_xscale('log')

		axDB.legend(loc=1, ncol=2 )

		#plot SCIMES




		fig.tight_layout()
		plt.savefig( "compareParaArea.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaArea.png"  ,  bbox_inches='tight',dpi=300)




	def totaFluxDistribution(self):
		"""
		Compare the change of molecular cloud numbers with
		:return:
		"""
		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)

		fig=plt.figure(figsize=(15,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,3,1)

		self.drawTotalFlux(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)
		at = AnchoredText(algDendro, loc=1, frameon=False)
		axDendro.add_artist(at)
		axDendro.set_ylabel(r"Total Flux (K km s$^-1$)")
		axDendro.set_xlabel(r"CO cutoff ($\sigma$)")
		axDendro.legend(loc=3)
		#plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro)
		self.drawTotalFlux(axDB,tb8DB,tb16DB, label8DB,  label16DB ,sigmaListDB)
		at = AnchoredText(algDB, loc=1, frameon=False)
		axDB.add_artist(at)

		axDB.set_ylabel(r"Total Flux (K km s$^-1$)")

		axDB.set_xlabel(r"CO cutoff ($\sigma$)")

		axDB.legend(loc=4)

		#plot SCIMES



		fig.tight_layout()
		plt.savefig( "compareParaTotalFlux.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaTotalFlux.png"  ,  bbox_inches='tight',dpi=300)


	def numberDistribution(self):
		"""
		Compare the change of molecular cloud numbers with
		:return:
		"""
		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)

		fig=plt.figure(figsize=(15,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,3,1)

		self.drawNumber(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)
		at = AnchoredText(algDendro, loc=1, frameon=False)
		axDendro.add_artist(at)
		axDendro.set_ylabel(r"Total number of trunks")
		axDendro.set_xlabel(r"CO cutoff ($\sigma$)")
		axDendro.legend(loc=3)
		#plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro)
		self.drawNumber(axDB,tb8DB,tb16DB, label8DB,  label16DB ,sigmaListDB)
		at = AnchoredText(algDB, loc=1, frameon=False)
		axDB.add_artist(at)

		axDB.set_ylabel(r"Total number of clusters")

		axDB.set_xlabel(r"CO cutoff ($\sigma$)")

		axDB.legend(loc=3)

		#plot SCIMES




		fig.tight_layout()
		plt.savefig( "compareParaNumber.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaNumber.png"  ,  bbox_inches='tight',dpi=300)


	def drawFlux(self,ax,tb8List,tb16List, label8,label16, sigmaListDen ):

		#areaEdges=np.linspace(0.25/3600.,150,10000)
		#areaCenter=self.getEdgeCenter( areaEdges )

		areaEdges=np.linspace(8,1e5,3000)
		areaCenter=self.getEdgeCenter( areaEdges )


		totalTB=tb8List+tb16List
		labelStr=label8+label16

		for i in range( len(totalTB) ):

			eachTB = totalTB[i]
			if "sum" not in  eachTB.colnames:
				#dendrogra
				sum=eachTB["flux"]/self.getSumToFluxFactor()*0.2 # K km/s

			else: #dbscan



				sum=eachTB["sum"]*0.2 # K km/s


			######
			#self.getAlphaWithMCMC(sum, minArea= 324*sigmaListDen[i]*0.2 , maxArea=None, physicalArea=True)





			binN,binEdges=np.histogram( sum , bins=areaEdges  )


			ax.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,label=labelStr[i] ,alpha= 0.5 )




	def drawArea(self,ax,tb8List,tb16List, label8,label16, sigmaListDen ):



		areaEdges=np.linspace(0.25/3600.,150,10000)
		areaCenter=self.getEdgeCenter( areaEdges )


		totalTB=tb8List+tb16List
		labelStr=label8+label16

		for i in range( len(totalTB) ):

			eachTB = totalTB[i]

			binN,binEdges=np.histogram(eachTB["area_exact"]/3600., bins=areaEdges  )


			ax.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,label=labelStr[i] ,alpha= 0.5 )



	def drawNumber(self,ax,tb8List,tb16List,label8,label16,sigmaListDen ):
		Nlist8Den=self.getNList(tb8List)
		Nlist16Den=self.getNList(tb16List)


		ax.plot(sigmaListDen,Nlist8Den,'o-',label="min\_nPix = 8",color="blue",lw=0.5)
		ax.plot(sigmaListDen,Nlist16Den,'o-',label="min\_nPix = 16",color="green", lw=0.5)




	def drawTotalFlux(self,ax,tb8List,tb16List,label8,label16,sigmaListDen ):
		Nlist8Den=self.getTotalFluxList(tb8List)
		#Nlist16Den=self.getTotalFluxList(tb16List)


		#print Nlist8Den
		#print Nlist16Den

		ax.plot(sigmaListDen,Nlist8Den,'o-',label="min\_nPix = 8",color="blue",lw=0.5)
		#ax.plot(sigmaListDen,Nlist16Den,'o-',label="min\_nPix = 16",color="green", lw=0.5)




	def getAlphaList(self,tbList, minArea=0.0225 ):
		# calculate alpha and  error for each alpha for each tb

		alphaList=[]
		errorList=[]

		for eachTB in tbList:

			area= eachTB["area_exact"]

			meanA,stdA=self.getAlphaWithMCMC(  area ,  minArea= minArea ,  maxArea=None , physicalArea=False )

			alphaList.append(meanA)
			errorList.append( stdA)

		return  alphaList,  errorList


	def drawAlpha(self,ax,tb8List,tb16List, label8, label16,sigmaListDen ): #


		#fitting alpha and draw

		alpha8List, alpha8ErrorList = self.getAlphaList(tb8List)
		#alpha16List, alpha16ErrorList = self.getAlphaList(tb16List)

		#ax.plot(sigmaListDen,alpha8List,'o-',label="MinPix = 8",color="blue", markersize= 3, lw=1)
		#ax.plot(sigmaListDen,alpha16List,'o-',label="MinPix = 16",color="green", lw=0.5, markersize= 2.5  ,  alpha=0.8 )

		ax.errorbar(sigmaListDen, alpha8List, yerr= alpha8ErrorList , c='b', marker='o', capsize=1.5, elinewidth=0.8, lw=1,label=r"min\_nPix = 16" )

		return alpha8List,alpha8ErrorList

	def areaAndNumberDistribution(self, algorithm="Dendrogram" ):
		"""
		#draw the area the
		:return:
		"""


		#first, get TBList

		tb8,tb16,label8,label16,sigmaList=self.getTBList(algorithm=algorithm)



		#need to constrain the minP and PeakN, PeakSigma=lower sigma cut + 3 sigma, as the minDelta,



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



		areaEdges=np.linspace(0,150,10000)
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

		axArea.set_xlim( [ 0.005,150 ] )


		if algorithm=="DBSCAN":
			axArea.set_ylim( [ 0.8,50000 ] )

		else:
			axArea.set_ylim( [ 0.8,10000 ] )





		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Bin number of trunks ")


		axArea.legend(ncol=2)


		#draw scimes

		scimesTB=Table.read("ClusterCat_3_16Ve20.fit")

		binN,binEdges=np.histogram(scimesTB["area_exact"]/3600., bins=areaEdges  )


		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  ,  color='red',  markersize=1, lw=0.8,label=r"Scimes,3.0$\sigma$, P16" ,alpha= 0.5 )

		at = AnchoredText("Red: SCIMES,3.0$\sigma$, P16", loc=4, frameon=False)
		#at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
		axArea.add_artist(at)






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



			#DbscanSigmaList= np.arange(2,6.5,0.5)
			DbscanSigmaList= np.arange(2,7.5,0.5)

			for sigmas in DbscanSigmaList:
				tbName= "G2650CO12DBCatS{:.1f}P{}Con2.fit".format(sigmas, minPix)
				ttt8=Table.read(tbName)
				ttt8=self.cleanDBTB(ttt8,pixN=8,minV=sigmas,minDelta=3)
				
				TBList.append(ttt8  )
				ttt16=ttt8[ttt8["pixN"]>=16]
				ttt16=self.cleanDBTB(ttt16,pixN=16,minV=sigmas,minDelta=3)


				TBList16.append(ttt16  )
				TBLabelsP8.append(  r"{:.1f}$\sigma$, P8".format( sigmas)   )
				TBLabelsP16.append( r"{:.1f}$\sigma$, P16".format( sigmas)   )

			


			return TBList,TBList16,TBLabelsP8,TBLabelsP16,DbscanSigmaList


		else:

			TBListP8=[]
			TBListP16=[]

			TBLabelsP8=[]
			TBLabelsP16=[]

			#dendroSigmaList=[2,2.5 , 3, 3.5, 4,4.5,5, 5.5, 6]

			dendroSigmaList=[2,2.5 , 3, 3.5, 4,4.5,5, 5.5, 6,6.5 ]

			for sigmas in dendroSigmaList:
				tbName8= "minV{}minP{}_dendroCatTrunk.fit".format(sigmas, 8)
				tbName16= "minV{}minP{}_dendroCatTrunk.fit".format(sigmas, 16)

				TBListP8.append(Table.read(tbName8)  )
				TBListP16.append(Table.read(tbName16)  )


				TBLabelsP8.append(  r"{:.1f}$\sigma$, P8".format( sigmas)   )
				TBLabelsP16.append( r"{:.1f}$\sigma$, P16".format( sigmas)   )



			return TBListP8,TBListP16,TBLabelsP8,TBLabelsP16,dendroSigmaList



	def setMinVandPeak(self,cloudLabelFITS,COFITS, peakSigma=3,minP=8):
		"""
		:param cloudLabelFITS:
		:param peakSigma:
		:return:
		"""
		#reject cloud that peak values area less than peakSigma, and total peak number are less then minP



		dataCluster, head= myFITS.readFITS(cloudLabelFITS)

		dataCO,headCO= myFITS.readFITS(COFITS)

		clusterIndex1D= np.where( dataCluster>0)
		clusterValue1D=  dataCluster[clusterIndex1D ]

		Z0,Y0,X0 = clusterIndex1D

		allClouds,counts=np.unique( clusterValue1D , return_counts=True)


		widgets = ['Fast Dendro WithDBSCAN: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),  ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

		pbar = ProgressBar(widgets=widgets, maxval=len(allClouds))
		pbar.start()


		for i in range( len(allClouds) ):


			pbar.update(i)

			cloudID =  allClouds[i]

			pixN=counts[i]
			cloudIndex=self.getIndices(Z0,Y0,X0,clusterValue1D,cloudID)

			if pixN<minP:#reject clouds

				dataCluster[cloudIndex]=0

				continue




			coValues=  dataCO[ cloudIndex ]

			if np.nanmax(coValues) <  peakSigma*self.rms: #reject
				cloudIndex=self.getIndices(Z0,Y0,X0,clusterValue1D,cloudID)
				dataCluster[cloudIndex]=0

				continue
		#relabelclouds
		pbar.finish()
		#dataCluster[dataCluster>0]=1
		s=generate_binary_structure(3,1)

		newDataCluster= dataCluster>0



		labeled_redo, num_features=label(newDataCluster, structure=s) #first label core, then expand, otherwise, the expanding would wrongly connected

		print "Total number of clouds? ",  num_features

		tbDendro=Table.read( "minV5minP8_dendroCatTrunk.fit" )
		print "The dendrogramN is ",len(tbDendro)

		#save the fits

		fits.writeto("relabelFastDendrominPeak{}_P{}.fits".format( peakSigma, minP ), labeled_redo, header=headCO,overwrite=True)


	def fastDendro(self,COFITS,minDelta=3,minV=3,minP=8):

		COData,COHead=myFITS.readFITS( COFITS)

		print np.max(COData)
		#first create dendrogram
		self.computeDBSCAN(COData,COHead, min_sigma=minV,min_pix=3,connectivity=1,region="fastDendroTest")

		dbFITS="fastDendroTestdbscanS{}P3Con1.fits".format(minV)



		self.setMinVandPeak(dbFITS,COFITS, peakSigma=minDelta+minV,minP=minP)

	def clearnDBAssign(self,DBLabelFITS,DBTableFile	,pixN=8,minDelta=3,minV=2  ):

		minPeak=(minV+minDelta)*self.rms
		saveName="DBCLEAN{}_{}Label.fits".format( minV, pixN )
		saveNameTB="DBCLEAN{}_{}TB.fit".format( minV, pixN )

		DBTable=Table.read( DBTableFile )

		dataCluster,headCluster=myFITS.readFITS(DBLabelFITS )

		clusterIndex1D= np.where( dataCluster>0 )
		clusterValue1D=  dataCluster[clusterIndex1D ]

		Z0,Y0,X0 = clusterIndex1D
		#cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, newID)

		emptyTB= Table( DBTable[0] )
		print "Cleaning DBSCAN table..."

		widgets = ['Recalculating cloud parameters: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),  ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

		pbar = ProgressBar(widgets=widgets, maxval=len(DBTable))
		pbar.start()

		indexRun=0
		for eachDBRow in DBTable:
			indexRun=indexRun+1
			pbar.update(indexRun)
			cloudID=  eachDBRow["_idx"]

			pixNCloud=int(  eachDBRow["pixN"]  )
			peakCloud= eachDBRow["peak"]

			if peakCloud < minPeak or pixNCloud< pixN : # set as zero

				cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, cloudID)
				dataCluster[cloudIndex] = 0

				continue
			emptyTB.add_row( eachDBRow  )
		pbar.finish()
		#save
		fits.writeto(saveName,dataCluster,header=headCluster,overwrite=True)

		emptyTB.write( saveNameTB,overwrite=True  )

		return saveName, saveNameTB

	def drawCloudMap(self,drawChannel=98):
		"""
		#draw small clouds to check if the are real...

		one color for DBSCAN
		one color for dendrogram,

		draw 2sigma, because they would provide the smallest area of clouds,

		:return:
		"""

		COFITS="G2650Local30.fits"

		data,head=myFITS.readFITS(COFITS)

		WCSCO=WCS(head)

		channelRawCO=data[drawChannel]

		DBLabelFITS = "G2650CO12dbscanS2.0P8Con2.fits"
		DBTableFile= "G2650CO12DBCatS2.0P8Con2.fit"
		drawDBSCANtb=Table.read("G2650CO12DBCatS2.0P8Con2.fit")


		relabelDB,newDBTable= self.clearnDBAssign( DBLabelFITS,DBTableFile	,pixN=16, minDelta=3, minV=2  )


		drawDBSCANtb=self.cleanDBTB(drawDBSCANtb,minDelta=3,minV=2,pixN=16)


		drawDBSCANtb=self.removeWrongEdges(drawDBSCANtb)


		drawDBSCANData,drawDBSCANHead = myFITS.readFITS("G2650CO12dbscanS2.0P8Con2.fits")





		channelDBSCAN = drawDBSCANData[drawChannel]

		dbClouds=np.unique(channelDBSCAN)



		drawDENDROtb=Table.read("minV2minP16_dendroCatTrunk.fit")

		drawDENDROtb=self.removeWrongEdges(drawDENDROtb)


		drawDENDROData,drawDENDROHead = myFITS.readFITS("minV2minP16_TrunkAsign.fits")

		channelDENDRO =  drawDENDROData[drawChannel]

		dendroClouds=np.unique(channelDENDRO)




		maximumArea= 144 *0.25 #arcmin^2

		#
		#print drawDBSCANtb.colnames



		fig = plt.figure(1, figsize=(8,4.5) )
		rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
		#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

		rc('text', usetex=True)

		axCO= pywcsgrid2.subplot(111, header=   WCSCO  )
		axCO.imshow(channelRawCO,origin='lower',cmap="bone",vmin=0 ,vmax=3,interpolation='none')


		#draw Dendrogram.............
		#the trunk assign of Dendrogrom is wong, the cloud0 ara all missed, so we ignore them

		runIndex=0

		for eachDRC in dendroClouds:

			if runIndex == 0:
				labelDendro = "Dendrogram"


			else:
				labelDendro = None


			eachDRC=int(eachDRC)
			if eachDRC==0:
				continue

			cRow=  drawDENDROtb[drawDENDROtb["_idx"]==eachDRC  ]

			area=cRow["area_exact"]

			if area>maximumArea:
				continue

			else:
				#draw


				if np.isnan(cRow["x_cen"] ):
					continue
				#print eachDRC



				axCO.scatter(cRow["x_cen"], cRow["y_cen"], s=13,facecolors='none',edgecolors='r', linewidths=0.3,  label= labelDendro  )
				runIndex=runIndex+1


		#draw DBSCAN.............
		runIndex=0
		for eachDBC in dbClouds:



			if runIndex == 0:
				labelDB = "DBSCAN"


			else:
				labelDB = None

			if eachDBC==0:
				continue

			cRow=  drawDBSCANtb[drawDBSCANtb["_idx"]==eachDBC  ]

			if len(cRow)==0: #may be edge sources
				continue

			area=cRow["area_exact"]

			if area>maximumArea:
				continue

			else:
				#draw


				if np.isnan(cRow["x_cen"] ):
					continue


				axCO["gal"].scatter(cRow["x_cen"], cRow["y_cen"]  , s=10,facecolors='none',edgecolors='b',linewidths=0.3, label= labelDB )
				runIndex=runIndex+1

		axCO.legend(loc=3)

		axCO.set_ticklabel_type("absdeg", "absdeg")
		axCO.axis[:].major_ticks.set_color("w")

		axCO.set_xlim( [1050, 1650] )
		axCO.set_ylim( [425,  865 ] )

		fig.tight_layout()
		plt.savefig("checkCloud.pdf", bbox_inches="tight")

		plt.savefig("checkCloud.png", bbox_inches="tight",dpi=600)


	def getLVFITSByDBMASK(self,DBlabel,CO12FITS,PVHeadTempFITS):
		dataDB, headDB = myFITS.readFITS( DBlabel )
		dataCO,headCO= myFITS.readFITS( CO12FITS )

		pvData,pvHead= myFITS.readFITS( PVHeadTempFITS )

		mask=dataDB>0
		mask=mask+0
		coMask=dataCO*mask
		Nz,Ny,Nx=dataCO.shape
		pvData=np.nansum(coMask, axis=1 )/Ny

		fits.writeto("G2650PV_DBMASK.fits",pvData,header=pvHead, overwrite=True)

	def cleanAllDBfits(self):

		# DbscanSigmaList= np.arange(2,6.5,0.5)
		DbscanSigmaList = np.arange(2, 7.5, 0.5)

		for sigmas in DbscanSigmaList:
			for minPix in [8,16]:

				tbName = "G2650CO12DBCatS{:.1f}P{}Con2.fit".format(sigmas, 8)
				fitsName = "G2650CO12dbscanS{:.1f}P{}Con2.fits".format(sigmas, 8 )
				self.clearnDBAssign( fitsName,tbName, pixN=minPix, minV=sigmas, minDelta= 3 )




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

ursaMajor=""

#veloicty distance, relation
# 13.46359868  4.24787753




if 1:
	doDBSCAN.totaFluxDistribution()


	#doDBSCAN.fluxDistribution()

	#doDBSCAN.alphaDistribution()

	#doDBSCAN.areaDistribution()
	#doDBSCAN.numberDistribution()

	sys.exit()




if 0:

	#clean

	doDBSCAN.cleanAllDBfits()
	sys.exit()


if 0:
	doDBSCAN.drawCloudMap(drawChannel= 50 )
	#doDBSCAN
	#doDBSCAN.drawAreaDistribute("ClusterCat_3_16Ve20.fit", region="scimes")
	#doDBSCAN.drawAreaDistribute("taurusDB3_8.fit" , region="scimes" )

	sys.exit()




if 0: #high Galacticlatitudes, ursa major
	doDBSCAN.rms=0.16
	coFITS12UM="/home/qzyan/WORK/projects/NewUrsaMajorPaper/OriginalFITS/myCut12CO.fits"
	COData,COHead=myFITS.readFITS( coFITS12UM)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=2, min_pix=8, connectivity=2, region="UMCO12")

	#doDBSCAN.getCatFromLabelArray(coFITS12UM,"UMCO12dbscanS2P8Con2.fits",doDBSCAN.TBModel,saveMarker="UMCO12_2_8")
	doDBSCAN.drawAreaDistribute("UMCO12_2_8.fit" , region="Taurus" )

	sys.exit()








if 1: #Taurus
	COData,COHead=myFITS.readFITS( TaurusCO12FITS)
	doDBSCAN.rms=0.3
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=3,min_pix=8,connectivity=2,region="Taurus")

	#doDBSCAN.getCatFromLabelArray(TaurusCO12FITS,"TaurusdbscanS3P8Con2.fits",doDBSCAN.TBModel,saveMarker="taurusDB3_8")
	#doDBSCAN.drawAreaDistribute("taurusDB3_8.fit" , region="Taurus" )

	


	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=1,min_pix=25,connectivity=3)

	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=6,min_pix=9,connectivity=2)

	sys.exit()


if 0:


	doDBSCAN.drawDBSCANNumber()
	doDBSCAN.drawDBSCANArea()



if 0: #dilation SCIMES

	#scimesFITS= "/home/qzyan/WORK/myDownloads/MWISPcloud/ClusterAsgn_ComplicateVe.fits"
	#rawFITS="/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"

	scimesFITS= "ClusterAsgn_3_16Ve20.fits"
	rawFITS= G2650CO12FITS  #"/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"

	doDBSCAN.myDilation( scimesFITS , rawFITS, saveName="G2650SCIMES_3_16Ve20", startSigma=15 )

	sys.exit()

if 0:
	doDBSCAN.getLVFITSByDBMASK( "G2650CO12dbscanS2.0P8Con2.fits", G2650CO12FITS, "/home/qzyan/WORK/myDownloads/testScimes/G2650PV.fits"  )




if 0: #test Fast Dendrogram


	doDBSCAN.fastDendro("testDendro.fits",minV=5,minP=8)


	sys.exit()

if 0: #test Fast Dendrogram
	COData,COHead=myFITS.readFITS( G2650CO12FITS)


	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=3,connectivity=1,region="G2650CO12DBDendro")

	#should peak sigma be larger?
	doDBSCAN.setMinVandPeak( "G2650CO12DBDendrodbscanS5P3Con1.fits" ,G2650CO12FITS,minP=8,peakSigma=8 )

	sys.exit()






if 0: # get catalog from extended fits


	doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650DisCloudVe20_extend.fits",doDBSCAN.TBModel,saveMarker="G2650CloudForDisCat")
	sys.exit()







if 0:
	#doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650CO12dbscanS2P16Con2.fits",doDBSCAN.TBModel, saveMarker="G2650CO12DBCatS2P16Con2" )
	for i in np.arange(2 ,8,0.5):
		savename="G2650CO12DBCatS{}P{}Con2".format(i,8)
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650CO12dbscanS{}P8Con2.fits".format(i),doDBSCAN.TBModel,saveMarker=savename)










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

	doDBSCAN.drawAreaDistribute("ClusterCat_3_16Ve20.fit" , region="scimes" )


	#doDBSCAN.drawAreaDistribute("minV3minP16_dendroCatTrunk.fit" , region="Perseus" )

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