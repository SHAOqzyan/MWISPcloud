
from progressbar import *
import numpy as np

class powerLaw1:

    def __init__(self):
        pass


    # fitting function
    def calPowerLawProbLogcomponent1(self ,theta, dataArraym ,minV, maxV):
        alpha1 = theta[0]  # Mt is the turnover mass of molecular cores

        beta1 = 1 - alpha1

        if maxV == None:
            # normalFactor1=beta1/( 0 -minV**beta1 )
            normalFactor1 = beta1 / (-minV ** beta1)

        else:
            normalFactor1 = (alpha1 - 1) * minV ** (alpha1 - 1)  # beta1/( maxV**beta1 -minV**beta1 )

        return len(dataArraym) * np.log(normalFactor1) - alpha1 * np.sum(np.log(dataArraym))




    def fitPowerLawWithMCMCcomponent1(self, dataArray, sampleN=1000, burn_in=100, minV=None, maxV=None, thin=15):
        """
        fit a power law with MCMC
        """
        # if minV==None or maxV==None:
        # minV= min(dataArray)
        # maxV= max(dataArray)

        mass  = dataArray

        print "The minimum and maximum masses are (in solar mass), in the MWISPcloud folder", minV, maxV

        np.random.seed()

        alpha1 = np.random.uniform(1, 5)  # np.random.exponential(1)

        theta = [alpha1]

        p0 = self.calPowerLawProbLogcomponent1(theta, mass, minV, maxV)

        sampleK = []
        sampleAlpha = [alpha1]
        widgets = ['MCMCSmapleSlope: ', Percentage(), ' ', Bar(marker='>', left='|', right='|'),
                   ' ', ETA(), ' ', FileTransferSpeed()]  # see docs for other options

        pbar = ProgressBar(widgets=widgets, maxval=sampleN + burn_in + 1)
        pbar.start()

        recordSample = []

        for i in range(100000):

            newAlpha1 = sampleAlpha[-1] + np.random.normal(0, 0.5)  # np.random.exponential(1)

            theta = [newAlpha1]

            p1 = self.calPowerLawProbLogcomponent1(theta, mass, minV, maxV)

            if np.isinf(p1):
                continue

            randomR = np.random.uniform(0, 1)

            if p1 >= p0 or p1 - p0 > np.log(randomR):
                p0 = p1;

                sampleAlpha.append(newAlpha1)



            else:
                sampleAlpha.append(sampleAlpha[-1])

            if i % thin == 0:
                recordSample.append(sampleAlpha[-1])

            pbar.update(len(recordSample))  # this adds a little symbol at each iteration

            if len(recordSample) > sampleN + burn_in:
                break
        pbar.finish()
        # print mean( sampleAlpha[burn_in:] ), np.std(sampleAlpha[burn_in:]  )
        return np.array(recordSample[burn_in:])

        # sample theta

    # sample theta


    def getAlphaWithMCMC(self, areaArray, minArea=0.03, maxArea=None, physicalArea=True, verbose=True, plotTest=False,
                         saveMark=""):
        """
        areaArray should be in square armin**2
        :param areaArray:
        :param minArea:
        :param maxArea:
        :return:
        """

        print "Fitting index with MCMC..."

        if not physicalArea:
            areaArray = areaArray / 3600.

        if maxArea != None:
            select = np.logical_and(areaArray > minArea, areaArray < maxArea)

        else:
            select = areaArray > minArea

        rawArea = areaArray[select]

        if verbose:
            print "Run first chain for {} molecular clouds.".format(len(rawArea))
        part1 = self.fitPowerLawWithMCMCcomponent1(rawArea, minV=minArea, maxV=maxArea)
        if verbose:
            print "Run second chain for {} molecular clouds.".format(len(rawArea))

        part2 = self.fitPowerLawWithMCMCcomponent1(rawArea, minV=minArea, maxV=maxArea)

        allSample = np.concatenate([part1, part2])

        # test plot
        if plotTest:
            fig = plt.figure(figsize=(12, 6))
            ax0 = fig.add_subplot(1, 1, 1)
            # fig, axs = plt.subplots(nrows=1, ncols=2,  figsize=(12,6),sharex=True)
            rc('text', usetex=True)
            rc('font', **{'family': 'sans-serif', 'size': 13, 'serif': ['Helvetica']})

            ax0.scatter(part1, part2, s=10)

            plt.savefig("mcmcSampleTest.pdf", bbox_inches='tight')
            aaaaaa

        meanAlpha = np.mean(allSample)
        stdAlpha = np.std(allSample, ddof=1)
        if verbose:
            print "Alpha Mean: {:.2f}; std: {:.2f}".format(meanAlpha, stdAlpha)

        return round(meanAlpha, 2), round(stdAlpha, 2)


#test
if 1:
    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a ** g, b ** g
        return (ag + (bg - ag) * r) ** (1. / g)

    #samples=np.random.power(-1.5)

    doPowerLaw=powerLaw1()

    samples= rndm(1,1000,-3.5,2000)

    doPowerLaw.getAlphaWithMCMC(samples,minArea=1,maxArea=100  )
