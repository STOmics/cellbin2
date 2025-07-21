const resultJson = {"sampleId": "SS200000135",
  "pipelineVersion": "SAW-v8.2.0",
  "isProteinReport": "true",
  "alertData":[
    {
      "metrics": "Mean Reads (Under Tissue) per bin (bin200)(Gene)",
      "value": "8568.12 (<50000)",
      "alert": "Warning",
      "note": "Dependent on tissue type, RNA quality, sequencing depth, and image processing results. If the image orientation and tissue detection were correct, then a low mean of reads might indicate low sequencing depth, library complexity, or poor quality."
    },
    {
      "metrics": "Valid PID Reads(Protein)",
      "value": "85.12% (<50000)",
      "alert": "Warning",
      "note": "A low fraction of valid PID reads that can be mapped to the protein database. Please check the input protein database."
    }
  ],
  "baseImageList": [
   
  ],
  "tissueseg": {
    "title": "Tissue Segmentation",
    "src": ""},
  "cellbin": {
    "gene": {
      "statistics": {
        "title": "Cell Bin Statistics",
        "msg": [
          {
            "title": "Cell Count",
            "content": "NUmber of cells"
          },
          {
            "title": "Mean/Median Cell Area",
            "content": "Mean/Median cell area (pixel)"
          },
          {
            "title": "Mean/Median Gene Type",
            "content": "Mean/Median gene types per cell"
          },
          {
            "title": "Mean/Median MID",
            "content": "Mean/Median MID count per cell"
          }
        ],
        "data": [
          {
            "cellCount": "0",
            "meanCellArea": "0",
            "medianCellArea": "0",
            "meanGeneType": "0",
            "medianGeneType": "0",
            "meanMID": "0",
            "medianMID": "0",
			"Fraction_cells":"0",
			"cell_to_tissue":"0",
			"datatype":"CellBin"
			},
            {
            "cellCount": "0",
            "meanCellArea": "0",
            "medianCellArea": "0",
			"meanGeneType": "0",
            "medianGeneType": "0",
            "meanMID": "0",
            "medianMID": "0",
			"Fraction_cells":"0",
			"cell_to_tissue":"0",
			"datatype":"Adjusted"},
        ]
      },
      "cb_distribution": {
		"title":"CellBin Distribution",
        "msg": ["Violin"],
        "data": [
          {
            "src": ".\\assets\\rna\\cellbin\\MIDCount.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\cellbin\\GeneType.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\cellbin\\CellArea.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          }]
      },
	  "ad_distribution": {
		"title":"Adjusted Distribution",
        "msg": ["Violin plots show the distribution of deduplicated MID count, gene types and cell area in each bin"],
        "data": [
          {
            "src": ".\\assets\\rna\\adjusted\\MIDCount.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\adjusted\\GeneType.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\adjusted\\CellArea.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          }]
      },
      "clustering": {
        "title": "Clustering",
        "msg": [
          "(left) Multiomics (Transcriptomics and Proteomics) Spatial clustering by Leiden algorithm",
          "(right) UMAP projection of RNA and Protein"
        ],
        "data": {
          "spatial": [],
          "umap": []
        }
      },
	  "expression": {
        "title": "Spatial Gene Expression Distribution",
        "msg": ["Spatial gene expression distribution plot shows MID count at each spot (bin10)-RNA"],
        "heatmapImageObj": {
          "allBinsColorBar": "",
          "allBins":"",
		  "tissueBinsColorBar": "",
		  "tissueBins": ""
        }
      },
	  "scatter_plot":{
		"title":"Scatter Plot ",
        "msg": ["scatter"],
		"data": [
          {
            "src": ".\\assets\\rna\\cellbin\\celldensity.png",
          },
		  {
			"src": ".\\assets\\rna\\cellbin\\MID_counts.png"
		  },
		  {
			"src": ".\\assets\\rna\\adjusted\\MID_counts.png"
		  }
		  ]
	  }
    },
    "protein": {
      "statistics": {
		  
        "title": "Cell Bin Statistics",
        "msg": [
          {
            "title": "Cell Count",
            "content": "NUmber of cells"
          },
          {
            "title": "Mean/Median Cell Area",
            "content": "Mean/Median cell area (pixel)"
          },
          {
            "title": "Mean/Median MID",
            "content": "Mean/Median MID count per cell"
          }
        ],
        "data": [
          {
            "cellCount": "-",
            "meanCellArea": "-",
            "medianCellArea": "-",
            "meanGeneType": "-",
            "medianGeneType": "-",
            "meanMID": "-",
            "medianMID": "-",
			"Fraction_cells":"-",
			"cell_to_tissue":"-",
			"datatype":"CellBin"
			},
            {
            "cellCount": "-",
            "meanCellArea": "-",
            "medianCellArea": "-",
			"meanGeneType": "-",
            "medianGeneType": "-",
            "meanMID": "-",
            "medianMID": "-",
			"Fraction_cells":"-",
			"cell_to_tissue":"-",
			"datatype":"Adjusted"},
        ]
      },
      "cb_distribution": {
		"title":"CellBin Distribution",
        "msg": ["Violin"],
        "data": [
          {
            "src": ".\\assets\\rna\\cellbin\\MIDCount.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\cellbin\\GeneType.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\cellbin\\CellArea.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          }]
      },
	  "ad_distribution": {
		"title":"Adjusted Distribution",
        "msg": ["Violin plots show the distribution of deduplicated MID count, gene types and cell area in each bin"],
        "data": [
          {
            "src": ".\\assets\\rna\\adjusted\\MIDCount.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\adjusted\\GeneType.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\rna\\adjusted\\CellArea.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          }]
      },
	  "distribution": {
        "msg": ["Violin plots show the distribution of deduplicated MID count and cell area in each bin"],
        "data": [
          {
            "src": ".\\assets\\protein\\cellbin\\MIDCount.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}
          },
          {
            "src": ".\\assets\\protein\\cellbin\\CellArea.png",
            "info": {
              "q1": "-",
              "q2": "-",
              "q3": "-",
              "min": "-",
              "max": "-"}

          }
        ]
      },
      "clustering": {
        "title": "Clustering",
        "msg": [
          "(left) Multiomics (Transcriptomics and Proteomics) Spatial clustering by Leiden algorithm",
          "(right) UMAP projection of RNA and Protein"
        ],
        "data": {
          "spatial": [],
          "umap": []
        }

      },
	  "expression": {
        "title": "Spatial Gene Expression Distribution",
        "msg": ["Spatial gene expression distribution plot shows MID count at each spot (bin10)-RNA"],
        "heatmapImageObj": {
          "allBinsColorBar": "",
          "allBins":"",
		  "tissueBinsColorBar": "",
		  "tissueBins": ""
        }
      },
	  "scatter_plot":{
		"title":"Scatter Plot ",
        "msg": ["scatter"],
		"data": [
          {
            "src": ".\\assets\\protein\\cellbin\\celldensity.png",
          },
		  {
			"src": ".\\assets\\protein\\cellbin\\MID_counts.png"
		  },
		  {
			"src": ".\\assets\\protein\\adjusted\\MID_counts.png"
		  }
		  ]
	  }
	  }
  },
  "image": {
	"image_num": 1 ,
	"summary":{
	  "title": "Image Summary",
	  "msg": [],
	  "data":[
	  {"label": "The number of Images",
	  "value": "3"},
	  {"label": "Image name 1",
	  "value": "xxx_ssDNA"},
	  {"label": "bits(M)/channel",
	   "value": "256/3"},
	  {"label": "Image name 2",
	  "value": "xxx_DAPI"},
	  {"label": "bits(M)/channel",
	  "value": "256/3"},
	  {"label": "Image name 3",
	  "value": "xxx_mIF"},
	  {"label": "bits(M)/channel",
	  "value": "256/3"},
	  {"label":"ImageSizeX(mm)",
	  "value": "10"
	  },
	  {"label":"ImageSizeY(mm)",
	  "value": "10"
	  }
	  ]
	  },
	"tissuecut": {
        "title": "Image Tissuecut Result",
        "msg": ["In bin10, compared the tissuecut mask with original image"],
        "heatmapImageObj": {
          "tissueBins": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhIAAAISCAYAAACZPSa/AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAC+F0lEQVR4nOzdd7hcVbnH8e9ae0+f00t6IYHQeyB0UAQBBUFs2LBgA8tV9FqwU8QGCIhYES+KIiiiiIAKKl16T0ICIfX0Nn32Xu/9Y+2cJICCAir4fp6Hm9yTk5k5k3H2mrXe9/eCUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppdSLnPl3PwCllHqhC97/c7G5FOe+/ue8afFi6BXazns38S/fo++x6kVPX+RKKbWR4MjzhbcWGJl7LqYp2HURZAzx1JCue0/k3G1+xTE3P4KJDBIKODDOIPg31PqKGgCZOVkkFIJZTVouOh536XH6fqtelPSFrZT6r5d56Rkiu82j/02nE9waI6XA/0EoECVvkymBGHBm8o1TAIwQD0aE7SESGEwsEGxYWIzfME5x5wLBZkLLFccjP3uHvu+qFxV9QSul/mtlXnqGyO7zGXztqXCzgabBFBwSgQlBynaT7zcwuUB44q8kv5dyjC34hUh9RQ1XE+or67S9rB0D/GCfBXxotx/pe6960dAXs1Lqv0ph91OlttsWjLzjawT9MW61xYSCRAaZSHYirCQ7EAYcfhei4JCyRcoxJlkorLfxQsIA0VCTsCvF+A3jpHpSFI4I6LzxCyy95UQ6XtnJL/ebw1vPPAwuOlbfg9ULXvjvfgBKKfV8K+x+qkQteQCivbdgbP/PwwMgzmBbHWQERq0/qgBwBupgig4ig9QMJu0XEk+3iAAIu1II0LpnC2vOXkvhoG7qK4bpeE07WMfRjy3lnS/L07zo+f7JlXr+6WpYKfWiVlh4ipS3m0tu825ec+C1fGv8eugzEApStZiUILXkCMMKMhFg8g6pJG+PacE0LJISTNMgKYHmRocaVkDAZPztbPymKnWHyfjbbp9zAsF3Rmj87oMGIL/nl6TvK7+Ahwy2Nab7kVOpPjIIP3yzvi+rFxR9wSqlXrSyB3xdeNVWDO74Ocy4I16ZwrbGYMANhZj2GBm3/ugi75CK3bSQ8kCh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eR77RbHQC0gjIy4G8t1lCGq/sRLX+yOSoxPGd7werr2d5LuKCIC8XmEw0kSTQLLWI+n0kH6G5JHOV1/pxYNzRzC/2nBuA0jaHfTDZfLNJQBkPBLn02Jpl30b0G6A0qFoa6w3QoFU4ZzI5IHXEz97B6X5le8d7Tz7rZpXy0VPRIm8UQUgZN87AuqcOzL5F7lzG0TtnLepPv5BTD/lD0h2DogHU8JEhi4HNLcNoHE1QD9YL0WqyP2U4bkXw4c6lA8u0Ln8Vd/zPaNx5pu1d9QYsVompgmhVUUHkdKuaV/M5dwG4Vcbzm0QSafP4Mu30Lr9RfzZC7/AcyduJsaEoDlSA9pCMmKdlfn2EhNXv5rsQz3Kt+4lbXd+YFGgSUCDoEM16GdIOSGu9gmD/B74CJ1z9wT/acG5DaR5xkUa04TBxDD5ljHKpw6Rj6+SzDVgMRB7Gf29S5Rml0mX2v/llEXzjIs0a9bIWnVirULo9pFBbhMbi20/kXBuA/Avcuc2oOYZF6mKQBDyWsWWeq2s2S9G/R/1NjTPuEg1CcRKCelnEATJI6tXvda/vzjnnHPuv9Y84yJtnnGR3tPvh3POOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnnHPOOeecc84555xzzjnnvs//B8RhxBZbCVBeAAAAAElFTkSuQmCC"
        }
      },
    "image1_qc": {
      "title": "Image 1 QC",
      "msg": [
        {
          "title": "Image QC version",
          "content": "Image QC version"
        },
        {
          "title": "QC Pass",
          "content": "Whether the image(s) passed image QC quality check"
        },
        {
          "title": "Trackline Score",
          "content": "Reference score for image clarity"
        }],
	  "data": [
        {
          "label": "Image QC version",
          "value": "4.0.0"},
        {
          "label": "QC Pass",
          "value": "Pass"},
        {
          "label": "Trackline Score",
          "value": "69"},
		  {
			"label":"QC score" , 
			"value":"69"
		  }]
    },
	"image1_registration": {
      "title": "Image 1 Registration",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        // {
        //   "title": "Manual ScaleX",
        //   "content": "The lateral scaling based on image center (manual-registration)"
        // },
        // {
        //   "title": "Manual ScaleY",
        //   "content": "The longitudinal scaling based on image center (manual-registration)"
        // },
        // {
        //   "title": "Manual Rotation",
        //   "content": "The rotation angle based on image center (manual-registration)"
        // },
        // {
        //   "title": "Matrix X Offset",
        //   "content": "Gene expression matrix offset in x direction by DNB numbers"
        // },
        // {
        //   "title": "Matrix Y Offset",
        //   "content": "Gene expression matrix offset in y direction by DNB numbers"
        // },
        // {
        //   "title": "Matrix Height",
        //   "content": "Gene expression matrix height"
        // },
        // {
        //   "title": "Matrix Width",
        //   "content": "Gene expression matrix width"
        // }
      ],
      "data": [
        {
          "label": "ScaleX",
          "value": "1.02"},
        {
          "label": "ScaleY",
          "value": "1.02"},
        {
          "label": "Rotation",
          "value": "-0.013"},
        {
          "label": "Flip",
          "value": "True"},
        {
          "label": "Image X Offset",
          "value": "257"},
        {
          "label": "Image Y Offset",
          "value": "-2,781.97"},
        {
          "label": "Counter Clockwise Rotation",
          "value": "180"},
      ]
    },
	"image1_matrix": {
      "title": "Image 1 Manual and Matrix",
      "msg": [
        // {
        //   "title": "ScaleX",
        //   "content": "The lateral scaling between image and template"
        // },
        // {
        //   "title": "ScaleY",
        //   "content": "The longitudinal scaling between image and template"
        // },
        // {
        //   "title": "Rotation",
        //   "content": "The rotation angle of the image relative to the template"
        // },
        // {
        //   "title": "Flip",
        //   "content": "Whether the image is flipped horizontally"
        // },
        // {
        //   "title": "Image X Offset",
        //   "content": "Offset between image and matrix in x direction"
        // },
        // {
        //   "title": "Image Y Offset",
        //   "content": "Offset between image and matrix in y direction"
        // },
        // {
        //   "title": "Counter Clockwise Rotation",
        //   "content": "Counter clockwise rotation angle"
        // },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        // {
        //   "title": "Matrix X Offset",
        //   "content": "Gene expression matrix offset in x direction by DNB numbers"
        // },
        // {
        //   "title": "Matrix Y Offset",
        //   "content": "Gene expression matrix offset in y direction by DNB numbers"
        // },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ]
    },
	"image1_cellseg": {
      "title": "Image 1 Cell Segmentation",
      "msg": [
        // {
        //   "title": "ScaleX",
        //   "content": "The lateral scaling between image and template"
        // },
        // {
        //   "title": "ScaleY",
        //   "content": "The longitudinal scaling between image and template"
        // },
        // {
        //   "title": "Rotation",
        //   "content": "The rotation angle of the image relative to the template"
        // },
        // {
        //   "title": "Flip",
        //   "content": "Whether the image is flipped horizontally"
        // },
        // {
        //   "title": "Image X Offset",
        //   "content": "Offset between image and matrix in x direction"
        // },
        // {
        //   "title": "Image Y Offset",
        //   "content": "Offset between image and matrix in y direction"
        // },
        // {
        //   "title": "Counter Clockwise Rotation",
        //   "content": "Counter clockwise rotation angle"
        // },
        // {
        //   "title": "Manual ScaleX",
        //   "content": "The lateral scaling based on image center (manual-registration)"
        // },
        // {
        //   "title": "Manual ScaleY",
        //   "content": "The longitudinal scaling based on image center (manual-registration)"
        // },
        // {
        //   "title": "Manual Rotation",
        //   "content": "The rotation angle based on image center (manual-registration)"
        // },
        // {
        //   "title": "Matrix X Offset",
        //   "content": "Gene expression matrix offset in x direction by DNB numbers"
        // },
        // {
        //   "title": "Matrix Y Offset",
        //   "content": "Gene expression matrix offset in y direction by DNB numbers"
        // },
        // {
        //   "title": "Matrix Height",
        //   "content": "Gene expression matrix height"
        // },
        // {
        //   "title": "Matrix Width",
        //   "content": "Gene expression matrix width"
        // }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ],
	  "cellseg1": "",
	  "cellseg2": "",
	  "cellseg3": "",
	  "cellseg4": "",
	  "cellseg5": "",
	"cell_intensity": {
    "title": "Cell Intensity",
    "src": ""}},
	"image1_clarity": {
    "title": "Clarity",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png"
  },
  	"image1_chipbox": {
    "title": "Chip Box",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png",
    "chipbox_part_image":{
      "chipbox_part_image1_src":"",
      "chipbox_part_image2_src":"",
      "chipbox_part_image3_src":"",
      "chipbox_part_image4_src":"",
    }
  },
    "image1_trackpoint": {
        "title": "Track Point",
        "src": ".\\assets\\rna\\adjusted\\MIDCount.png",
        "small_chip_image": {
            "chip_image1_src":"",
            "chip_image2_src":"",
            "chip_image3_src":"",
            "chip_image4_src":""
        },
        "small_tissue_image": {
            "tissue_image1_src":"",
            "tissue_image2_src":"",
            "tissue_image3_src":"",
            "tissue_image4_src":""
        }
  },
    "image2_qc": {
      "title": "Image 1 QC",
      "msg": [
        {
          "title": "Image QC version",
          "content": "Image QC version"
        },
        {
          "title": "QC Pass",
          "content": "Whether the image(s) passed image QC quality check"
        },
        {
          "title": "Trackline Score",
          "content": "Reference score for image clarity"
        }],
	  "data": [
        {
          "label": "Image QC version",
          "value": "4.0.0"},
        {
          "label": "QC Pass",
          "value": "Pass"},
        {
          "label": "Trackline Score",
          "value": "69"},
		  {
			"label":"QC score" , 
			"value":"69"
		  }]
    },
	"image2_registration": {
      "title": "Image 2 Registration",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "ScaleX",
          "value": "1.02"},
        {
          "label": "ScaleY",
          "value": "1.02"},
        {
          "label": "Rotation",
          "value": "-0.013"},
        {
          "label": "Flip",
          "value": "True"},
        {
          "label": "Image X Offset",
          "value": "257"},
        {
          "label": "Image Y Offset",
          "value": "-2,781.97"},
        {
          "label": "Counter Clockwise Rotation",
          "value": "180"},
      ]
    },
	"image2_matrix": {
      "title": "Image 2 Manual and Matrix",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ]
    },
	"image2_cellseg": {
      "title": "Image 2 Cell Segmentation",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ],
	  "cellseg1": "",
	  "cellseg2": "",
	  "cellseg3": "",
	  "cellseg4": "",
	  "cellseg5": "",
	"cell_intensity": {
    "title": "Cell Intensity",
    "src": ""}},
	"image2_clarity": {
    "title": "Clarity",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png"
  },
  	"image2_chipbox": {
    "title": "Chip Box",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png",
    "chipbox_part_image":{
      "chipbox_part_image1_src":"",
      "chipbox_part_image2_src":"",
      "chipbox_part_image3_src":"",
      "chipbox_part_image4_src":"",
      }
    },
    "image2_trackpoint": {
    "title": "Track Point",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png",
	"small_chip_image": {
            "chip_image1_src":"",
            "chip_image2_src":"",
            "chip_image3_src":"",
            "chip_image4_src":""
        },
        "small_tissue_image": {
            "tissue_image1_src":"",
            "tissue_image2_src":"",
            "tissue_image3_src":"",
            "tissue_image4_src":""
        }
  },
   "image3_qc": {
      "title": "Image 2 QC",
      "msg": [
        {
          "title": "Image QC version",
          "content": "Image QC version"
        },
        {
          "title": "QC Pass",
          "content": "Whether the image(s) passed image QC quality check"
        },
        {
          "title": "Trackline Score",
          "content": "Reference score for image clarity"
        }],
	  "data": [
        {
          "label": "Image QC version",
          "value": "4.0.0"},
        {
          "label": "QC Pass",
          "value": "Pass"},
        {
          "label": "Trackline Score",
          "value": "69"},
		  {
			"label":"QC score" , 
			"value":"69"
		  }]
    },
	"image3_registration": {
      "title": "Image 3 Registration",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "ScaleX",
          "value": "1.02"},
        {
          "label": "ScaleY",
          "value": "1.02"},
        {
          "label": "Rotation",
          "value": "-0.013"},
        {
          "label": "Flip",
          "value": "True"},
        {
          "label": "Image X Offset",
          "value": "257"},
        {
          "label": "Image Y Offset",
          "value": "-2,781.97"},
        {
          "label": "Counter Clockwise Rotation",
          "value": "180"},
      ]
    },
	"image3_matrix": {
      "title": "Image 3 Manual and Matrix",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ]
    },
	"image3_cellseg": {
      "title": "Image 3 Cell Segmentation",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ],
	  "cellseg1": "",
	  "cellseg2": "",
	  "cellseg3": "",
	  "cellseg4": "",
	  "cellseg5": "",
	"cell_intensity": {
    "title": "Cell Intensity",
    "src": ""}},
	"image3_clarity": {
    "title": "Clarity",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png"
  },
  	"image3_chipbox": {
      "title": "Chip Box",
      "src": ".\\assets\\rna\\adjusted\\MIDCount.png",
      "chipbox_part_image": {
        "chipbox_part_image1_src": "",
        "chipbox_part_image2_src": "",
        "chipbox_part_image3_src": "",
        "chipbox_part_image4_src": "",
      }
    },
    "image3_trackpoint": {
    "title": "Track Point",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png",
	"small_chip_image": {
            "chip_image1_src":"",
            "chip_image2_src":"",
            "chip_image3_src":"",
            "chip_image4_src":""
        },
        "small_tissue_image": {
            "tissue_image1_src":"",
            "tissue_image2_src":"",
            "tissue_image3_src":"",
            "tissue_image4_src":""
        }
  },
  "image4_qc": {
      "title": "Image 4 QC",
      "msg": [
        {
          "title": "Image QC version",
          "content": "Image QC version"
        },
        {
          "title": "QC Pass",
          "content": "Whether the image(s) passed image QC quality check"
        },
        {
          "title": "Trackline Score",
          "content": "Reference score for image clarity"
        }],
	  "data": [
        {
          "label": "Image QC version",
          "value": "4.0.0"},
        {
          "label": "QC Pass",
          "value": "Pass"},
        {
          "label": "Trackline Score",
          "value": "69"},
		  {
			"label":"QC score" , 
			"value":"69"
		  }]
    },
	"image4_registration": {
      "title": "Image 4 Registration",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "ScaleX",
          "value": "1.02"},
        {
          "label": "ScaleY",
          "value": "1.02"},
        {
          "label": "Rotation",
          "value": "-0.013"},
        {
          "label": "Flip",
          "value": "True"},
        {
          "label": "Image X Offset",
          "value": "257"},
        {
          "label": "Image Y Offset",
          "value": "-2,781.97"},
        {
          "label": "Counter Clockwise Rotation",
          "value": "180"},
      ]
    },
	"image4_matrix": {
      "title": "Image 4 Manual and Matrix",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ]
    },
	"image4_cellseg": {
      "title": "Image 4 Cell Segmentation",
      "title": "Image 4 Cell Segmentation",
      "msg": [
        {
          "title": "ScaleX",
          "content": "The lateral scaling between image and template"
        },
        {
          "title": "ScaleY",
          "content": "The longitudinal scaling between image and template"
        },
        {
          "title": "Rotation",
          "content": "The rotation angle of the image relative to the template"
        },
        {
          "title": "Flip",
          "content": "Whether the image is flipped horizontally"
        },
        {
          "title": "Image X Offset",
          "content": "Offset between image and matrix in x direction"
        },
        {
          "title": "Image Y Offset",
          "content": "Offset between image and matrix in y direction"
        },
        {
          "title": "Counter Clockwise Rotation",
          "content": "Counter clockwise rotation angle"
        },
        {
          "title": "Manual ScaleX",
          "content": "The lateral scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual ScaleY",
          "content": "The longitudinal scaling based on image center (manual-registration)"
        },
        {
          "title": "Manual Rotation",
          "content": "The rotation angle based on image center (manual-registration)"
        },
        {
          "title": "Matrix X Offset",
          "content": "Gene expression matrix offset in x direction by DNB numbers"
        },
        {
          "title": "Matrix Y Offset",
          "content": "Gene expression matrix offset in y direction by DNB numbers"
        },
        {
          "title": "Matrix Height",
          "content": "Gene expression matrix height"
        },
        {
          "title": "Matrix Width",
          "content": "Gene expression matrix width"
        }
      ],
      "data": [
        {
          "label": "Manual scaleX",
          "value": "-"},
        {
          "label": "Manual scaleY",
          "value": "-"},
        {
          "label": "Manual Rotation",
          "value": "-"},
        {
          "label": "Matrix X Offset",
          "value": "0.00"},
        {
          "label": "Matrix Y Offset",
          "value": "0.00"},
        {
          "label": "Matrix Height",
          "value": "14695"},
        {
          "label": "Matrix Width",
          "value": "14695"},
      ],
	  "cellseg1": "",
	  "cellseg2": "",
	  "cellseg3": "",
	  "cellseg4": "",
	  "cellseg5": "",
	"cell_intensity": {
    "title": "Cell Intensity",
    "src": ""}},
	"image4_clarity": {
    "title": "Clarity",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png"
  },
  	"image4_chipbox": {
    "title": "Chip Box",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png",
    "chipbox_part_image": {
        "chipbox_part_image1_src": "",
        "chipbox_part_image2_src": "",
        "chipbox_part_image3_src": "",
        "chipbox_part_image4_src": ""
      }
    },
    "image4_trackpoint": {
    "title": "Track Point",
	"src": ".\\assets\\rna\\adjusted\\MIDCount.png",
	"small_chip_image": {
            "chip_image1_src":"",
            "chip_image2_src":"",
            "chip_image3_src":"",
            "chip_image4_src":""
        },
        "small_tissue_image": {
            "tissue_image1_src":"",
            "tissue_image2_src":"",
            "tissue_image3_src":"",
            "tissue_image4_src":""
        }
  },
}
}