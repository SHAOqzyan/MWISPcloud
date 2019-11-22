
import os
import numpy as np
from astropy.table import Table
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

from scipy.ndimage import label, generate_binary_structure,binary_erosion,binary_dilation

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



	def computeDBSCAN(self,COdata,COHead, min_sigma=2, min_pix=7, connectivity=1 ,region=""  ):
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

		COdata[COdata<minValue]=0

		Nz,Ny,Nx  = COdata.shape
		extendMask = np.zeros([Nz+2,Ny+2,Nx+2] ,dtype=int)

		extendMask[1:-1,1:-1,1:-1][COdata>=minValue]=1

		s=generate_binary_structure(3,connectivity)

		if connectivity==1:
			coreArray=self.sumEdgeByCon1(extendMask)

		if connectivity==2:
			coreArray=self.sumEdgeByCon2(extendMask)

		if connectivity==3:
			coreArray=self.sumEdgeByCon3(extendMask)

		coreArray = coreArray>=min_pix
		coreArray[COdata<minValue ]=False #remove falsely  core values

		coreArray=coreArray+0

		allArray=  binary_dilation(coreArray,structure=s) #expand the corepoints according to the structure

		allArray[COdata<minValue ]=False #remove falsely expanded values
		allArray=allArray+0

		if 1:
			labeled_array,num_features=label(allArray,structure=s)
			saveName="{}dbscanS{}P{}Con{}.fits".format( region,min_sigma,min_pix,connectivity )


		else:#no expand,only core array, just for test
			labeled_array,num_features=label(coreArray,structure=s)
			saveName="{}dbscanS{}P{}Con{}NoExpand.fits".format(region, min_sigma,min_pix,connectivity )




		print num_features,"features found!"

		fits.writeto(saveName, labeled_array, header=COHead, overwrite=True)




	def maskByGrow(self,COFITS,peakSigma=3,minV=1.):

		COData,COHead=myFITS.readFITS( COFITS )
		markers=np.zeros_like(COData )

		COData[COData<minV* self.rms]=0

		markers[COData>peakSigma*self.rms] = 1

		labels=watershed(COData,markers)
		fits.writeto("growMaskPeak3Min1.fits",labels,header=COHead,overwrite=True)






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

		minPeak = peakSigma*self.rms

		saveName=saveMarker+"Sigma{}_P{}FastDBSCAN.fit".format(rms,minPix)

		clusterTBOld=Table.read( TBModel )

		###
		dataCO, headCO = myFITS.readFITS( CO12FITS )

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

		ids,count=np.unique(dataCluster,return_counts=True )

		GoodIDs=  ids[count>=minPix ]

		GoodCount = count[ count>=minPix  ]



		GoodCount = GoodCount[ GoodIDs>0  ]

		GoodIDs=  GoodIDs[GoodIDs>0 ]


		print "Total number of turnks? ",len(ids)
		print "Total number of Good Trunks? ",len(GoodIDs)

		#dataCO,headCO=doFITS.readFITS( CO12FITS )
		widgets = ['Recalculating cloud parameters: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),  ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

		pbar = ProgressBar(widgets=widgets, maxval=len(GoodIDs))
		pbar.start()


		catTB=newTB.copy()



		runIndex=0

		for i in  range(len(GoodIDs)) :
			if i==0:
				continue

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

		TB2_16= "DBSCAN2_16Sigma1_P1FastDBSCAN.fit"
		#TB2_16= "DBSCAN2_9Sigma1_P1FastDBSCAN.fit"
		TB25_9="DBSCAN25_9Sigma1_P1FastDBSCAN.fit"
		TB35_9="DBSCAN35_9Sigma1_P1FastDBSCAN.fit"
		TB45_9="DBSCAN45_9Sigma1_P1FastDBSCAN.fit"
		TB55_9="DBSCAN55_9Sigma1_P1FastDBSCAN.fit"
		TB65_9="DBSCAN65_9Sigma1_P1FastDBSCAN.fit"
		TB75_9="DBSCAN75_9Sigma1_P1FastDBSCAN.fit"

		TB3_9= "DBSCAN3_9Sigma1_P1FastDBSCAN.fit"
		TB4_9= "DBSCAN4_9Sigma1_P1FastDBSCAN.fit"
		TB5_9= "DBSCAN5_9Sigma1_P1FastDBSCAN.fit"
		TB6_9= "DBSCAN6_9Sigma1_P1FastDBSCAN.fit"
		TB7_9= "DBSCAN7_9Sigma1_P1FastDBSCAN.fit"

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

		TB2_16= "DBSCAN2_16Sigma1_P1FastDBSCAN.fit"
		#TB2_16= "DBSCAN2_9Sigma1_P1FastDBSCAN.fit"
		TB25_9="DBSCAN25_9Sigma1_P1FastDBSCAN.fit"
		TB35_9="DBSCAN35_9Sigma1_P1FastDBSCAN.fit"
		TB45_9="DBSCAN45_9Sigma1_P1FastDBSCAN.fit"
		TB55_9="DBSCAN55_9Sigma1_P1FastDBSCAN.fit"
		TB65_9="DBSCAN65_9Sigma1_P1FastDBSCAN.fit"
		TB75_9="DBSCAN75_9Sigma1_P1FastDBSCAN.fit"

		TB3_9= "DBSCAN3_9Sigma1_P1FastDBSCAN.fit"
		TB4_9= "DBSCAN4_9Sigma1_P1FastDBSCAN.fit"
		TB5_9= "DBSCAN5_9Sigma1_P1FastDBSCAN.fit"
		TB6_9= "DBSCAN6_9Sigma1_P1FastDBSCAN.fit"
		TB7_9= "DBSCAN7_9Sigma1_P1FastDBSCAN.fit"

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



	def ZZ(self):
		pass


doDBSCAN=myDBSCAN()

CO12FITS="/home/qzyan/WORK/myDownloads/testFellwalker/WMSIPDBSCAN/G2650Local30.fits"
DBMaskFITS= "/home/qzyan/WORK/myDownloads/testFellwalker/G2650DB_1_25.fits"
TaurusCO12FITS="/home/qzyan/WORK/dataDisk/Taurus/t12_new.fits"


if 1:#Example
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
	COData,COHead=myFITS.readFITS( CO12FITS)

	for i in np.arange(2,8,0.5):
		doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=i,min_pix=9,connectivity=2)

	sys.exit()



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

if 1:
	COData,COHead=myFITS.readFITS( CO12FITS)

	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=1,min_pix=25,connectivity=3)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=2,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=3,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=6,min_pix=9,connectivity=2)



if 0:
	#doDBSCAN.maskByGrow(CO12FITS)
	#doDBSCAN.drawTrueArea()
	doDBSCAN.drawDBSCANNumber()
	doDBSCAN.drawDBSCANArea()





if 0:

	ModelTB="minV3minP16_dendroCatTrunk.fit"

	doDBSCAN.getCatFromLabelArray(CO12FITS,DBMaskFITS,  ModelTB,  rms=1,minPix=25 )