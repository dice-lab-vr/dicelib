import os, sys
import numpy as np
from . import ui

from nibabel.streamlines import load, save, Tractogram

from dipy.segment.clustering import QuickBundles# , qbx_and_merge
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
# from dipy.tracking.streamline import Streamlines

def load_connectome(filename_cm):
    try:
        connectome = np.loadtxt( filename_cm, delimiter=',' )
    except:
        connectome = np.loadtxt( filename_cm, delimiter=' ' )
    return connectome

def create_structure_2_levels(filename_trk, filename_cm, filename_fs, output_folder, QB_threshold=10, suffix='', metric=None, verbose=0, force=False  ):
    ''' Creates a hierarchical structure with 2 levels to use in COMMIT2 
    Input:
        filename_trk: Path to the tractogram  
        filename_cm: Path to the connectome 
        filename_fs: Path to the assignments file of the streamlines  
        output_folder: Output path 
        QB_threshold: Threshold to use in QuickBundles 
        suffix: Sufix to name the output files 
        metric: Metric to use in QuickBundles   
        verbose: What information to print (must be in [0...4] as defined in ui)
        force: Force overwriting of the output

    Output:
        In the output_folder it will create the files:
            structureIC_level1: First level of the structure 
            structureIC_level2: Second level of the structure
            structureIC_level1_level2: Hierarchical structure 
    '''

    ui.set_verbose( verbose )
    

    if metric==None:
        feature = ResampleFeature(nb_points=20)
        metric = AveragePointwiseEuclideanMetric(feature)
    
    output_l1 = os.path.join( output_folder , 'structureIC_level1.npy')
    output_l2 = os.path.join( output_folder , 'structureIC_level2.npy'+suffix)
    output_l1_l1 = os.path.join( output_folder , 'structureIC_level1_level2.npy'+suffix)

    # Structures for streamlines 
    connStr_p = []      # Parcellation (bundles)
    connStr_c = []      # Clusters (of bundles)
    connStr_pc = []     # Parcellation and clusters
    clusterCentroids = []
    
    # 0. Check files 
    ui.INFO( f'Checking files' )

    for e in [filename_trk, filename_cm, filename_fs]:
        if not os.path.isfile(e):
            print('File not foud: %s'%e)
            ui.INFO( f'File "{e}" not found' )
            sys.exit(1)

    if os.path.isfile(output_l1_l1) and not force:
        ui.ERROR(f"Output structure {output_l1_l1} already exists, use -f to overwrite")
    
    # 1. Compute/load the connectome 
    connectome          = load_connectome( filename_cm )
    fibers_assignment   = np.loadtxt( filename_fs )
    streamlines_input   = load( filename_trk )
    n1, n2 = connectome.shape

    connStrMat          = [[{'indxFibeAssignment':[],'numClusters':0, 'numIndxPerCluster':[], 'indxPerCluster':[], 'isVB':0, 'isVB_gt':0} for j in range(n2+1)] for i in range(n1+1)] # connectivityStructure

    # 2. Kept only streamlines connected to ROIs
    ui.INFO( f'Looking for streamlines connected to ROIs' )
    numFibersToDiscard = 0
    streams = []
    for r in range(fibers_assignment.shape[0]):            
        i = int(fibers_assignment[r,0])
        j = int(fibers_assignment[r,1])
        if i!=0 and j!=0:
            if j<i:
                tmp = j
                j = i 
                i = tmp
            connStrMat[i][j]['indxFibeAssignment'].append(r)  # "Connectivity matrix with idxs of streamlines in each connection"
        else:
            numFibersToDiscard += 1

    # TODO: The non-connected streamlines will not be consider in the groups, how we should handle them
    if numFibersToDiscard > 0:
        ui.WARNING(f'Number streamlines discarded: {numFibersToDiscard}')
    else:
        ui.INFO(f'Number streamlines discarded: {numFibersToDiscard}')

    # 3. Cluster the streamlines in each connection 
    ui.INFO( f'Creating clusters' )

    lst_numStreamlines          = []
    lst_numClusters             = []
    lst_streamlinesPerCluster   = []

    tot = 0

    for i in range(n1):
        for j in range(i,n2):
            if connectome[i,j] > 0 :

                np.random.seed( 0 )
                tot += 1

                streamlines = [streamlines_input.streamlines[k] for k in connStrMat[i+1][j+1]['indxFibeAssignment']]
                
                numStreamlines = len(streamlines)
                qb = QuickBundles( threshold = QB_threshold, metric=metric )
                clusters = qb.cluster(streamlines)

                numClusters = len(clusters) 

                # print('C %d - i %d, j %d numStreamlines %d numClusters  %d %d'%(connectome[i,j],i+1, j+1, numStreamlines,  numClusters,  numStreamlines/numClusters))
                lst_numStreamlines.append(numStreamlines)
                lst_numClusters.append(numClusters)
                lst_streamlinesPerCluster.append(numStreamlines/numClusters)

                indxPerCluster = []
                numIndxPerCluster = []
                for cont, cluster in enumerate(clusters):
                    idxTmp = []
                    for idx in cluster.indices:
                        idxTmp.append(connStrMat[i+1][j+1]['indxFibeAssignment'][idx])        
                    indxPerCluster.append(np.array(idxTmp))
                    connStr_c.append(np.array(idxTmp))
                    connStr_pc.append(np.array(idxTmp))
                    
                    numIndxPerCluster.append(len(cluster.indices))

                    ff= (cluster.centroid)
                    clusterCentroids.append(ff)

                connStr_pc.append(np.array(connStrMat[i+1][j+1]['indxFibeAssignment']))

                connStrMat[i+1][j+1]['numIndxPerCluster'] = numIndxPerCluster
                connStrMat[i+1][j+1]['indxPerCluster'] = indxPerCluster
                connStrMat[i+1][j+1]['numClusters'] = numClusters

                connStr_p.append(np.array(connStrMat[i+1][j+1]['indxFibeAssignment']))

    np.save( output_l1, np.array( connStr_p, dtype=object))
    np.save( output_l2, np.array( connStr_c, dtype=object ))
    np.save( output_l1_l1, np.array( connStr_pc, dtype=object ))
    