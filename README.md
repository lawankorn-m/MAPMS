# MAPMS 


1) Single Feature Extractions

1.1 Descriptors (directly from SMILES or Peptide)

We extract classical molecular descriptors directly from SMILES representations:

TPC_Feature = TPC(Peptide), 

MACCS_Feature = MACCS(SMILES), 

PubChem_Feature = PubChem(SMILES), 

ERG_Feature = ERG(SMILES), 

ECFP_Feature = ECFP(SMILES)

These descriptors provide complementary structural and chemical information. While individually standard in cheminformatics, they form the basis for integrating multiple modalities.

1.2 Feature Extractor from Pretrained Models

We leverage pretrained embedding models to capture richer representations from SMILES:

ESM_Embedding = ESM(SMILES), 

Mol2Vec_Embedding = Mol2Vec(SMILES)

The pretrained embeddings introduce semantic and context-aware molecular features, complementing classical descriptors and providing novel multi-view representations when combined in downstream tasks.
________________________________________

2) Multimodalities
To exploit different feature types, we construct separate branches tailored to each modality:

Embed1 = CNN(TPC_Feature),

Embed2 = BiGRU(concat(TPC_Feature, Mol2Vec)),

Embed3 = CNN(concat(TPC_Feature, ESM)),

Embed4 = MLP(concat(MACCS_Feature, PubChem_Feature, ERG_Feature, ECFP_Feature))


Embed1 captures structural patterns directly from TPC descriptors using CNN. 
Embed2 and Embed3 demonstrate the novel integration of classical descriptors with pretrained embeddings, allowing synergistic feature learning. 
Embed4 encodes multiple classical descriptors using MLP, ensuring that complementary chemical fingerprints are effectively modeled. 
This multi-branch design represents the core novelty of our architecture, combining heterogeneous molecular representations in a unified framework.
________________________________________
3) Multimodal Feature Representation

Fusion_Embedding = MLP(concat(Embed1, Embed2, Embed3, Embed4))

All branch embeddings are fused via concatenation and processed through an MLP, forming a comprehensive multimodal molecular representation. This fusion allows the model to exploit complementary strengths of each feature type, improving the overall expressive power.
________________________________________
4) Classification

y = DNN(Fusion_Embedding)

The fused embedding is fed into a DNN for prediction of molecular properties. By leveraging multimodal representations, the classifier benefits from richer, synergistic feature information, which is difficult to capture using single-feature approaches.

