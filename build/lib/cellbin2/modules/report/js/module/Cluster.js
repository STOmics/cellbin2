
const Cluster = {

  props: {
    moduleTitle: String,
    msg: String,
    data: Array,
    prefix: String,
    binSize: String,
    baseSrc: String
  },
  setup(props) {
    const { prefix } = props;
    const spatialOpacity = ref(1);
    const ifShowExplain = ref(false);

    const spatialPrefix = prefix + '-spatial-';
    const umapPrefix = prefix + '-umap-';

    const layout = {
      yaxis: { autorange: 'reversed', scaleanchor: "x", scaleratio: 1, showgrid: false, showticklabels: false, zeroline: false },
      xaxis: { showgrid: false, showticklabels: false, zeroline: false },
      margin: { 'l': 5, 't': 10, 'r': 10, 'b': 10 },
      dragmode: false,
      legend: {
        itemsizing: 'constant',
        itemwidth: 10,
        x: 1

      }
    };

    const config = {
      scrollZoom: false,
      displayModeBar: false,
      displaylogo: false,
    };


    function changeSpatialImageOpacity(prefix, opacity) {
      spatialOpacity.value = opacity;
      Plotly.restyle(prefix + 'plotCanvas', {
        'marker.opacity': opacity,
      });
    };
    function switchOpaticy() {
      spatialOpacity.value = spatialOpacity.value == 1 ? 0 : 1;
      changeSpatialImageOpacity(spatialPrefix, spatialOpacity.value);
    };

    onMounted(() => {
      console.log('Cluster Module mounted, props: ', props);
      Plotly.newPlot(spatialPrefix + 'plotCanvas', props.data.spatial, { ...layout, images: [props.baseSrc] }, config);
      Plotly.newPlot(umapPrefix + 'plotCanvas', props.data.umap, layout, config);
    });

    return {
      props,
      spatialOpacity,
      spatialPrefix,
      umapPrefix,
      ifShowExplain,
      switchOpaticy,
      changeSpatialImageOpacity
    };
  },
  template: `
  <div class="module-box" style="width: 1200px;">
    <div class="module-title-box" >
      <div class="title-box-left">
        <span class="title-icon"></span>
        <span class="title-label">{{props.moduleTitle}}</span>
        <el-button v-if="props.msg && props.msg.length" circle type="info" plain size="small" >
          <el-icon @click="ifShowExplain = !ifShowExplain" size="24">
            <zhrAsk />
          </el-icon>
        </el-button>
      </div>
    </div>

    <div class="module-msg" v-if="ifShowExplain">
      <div v-for="item in props.msg">
      {{ item }}
      </div> 
    </div>

    <div class="module-content-box" style="display: flex;justify-content: space-between;height: 490px;">
      <div style="width: 560px;height: 100%; border: solid 1px #f2f2f2; border-radius: 5px;">

              <div class="cluster-container" style="width: 100%;height: 450px;display:flex;align-items: center;"> 
                    
                <div style="width: 40px;height: 200px; position: relative;display: flex; align-items: center; justify-content: center;flex-direction: column;margin-left: 5px;">
                  <el-slider :vertical="true" v-model="spatialOpacity" :min='0' height="180px" :max="1" step=0.01 @change="changeSpatialImageOpacity(spatialPrefix, spatialOpacity)" :format-tooltip="formatTooltip" :show-tooltip="false">
                  </el-slider>
                  <el-button @click="switchOpaticy()" :icon="Switch" plain size="normal" style="margin: 20px 0;padding: 3px;border-radius: 15px;">
                    <el-icon><Switch /></el-icon>
                  </el-button>
                </div>
                <div :id="spatialPrefix + 'plotCanvas'" style="width:100%;height:100%;">
                </div>

              
              </div>
              <div style="font-size: 18px; font-weight: 600;text-align: center; height: calc(100% - 450px);">
                Tissue Plot with Spots ({{props.binSize}})
              </div>
      </div>




      <div style="width: 560px;height: 100%; border: solid 1px #f2f2f2; border-radius: 5px;">
              <div class="cluster-container" style="width: 100%;height: 450px;display:flex;align-items: center;">
                 <div :id="umapPrefix + 'plotCanvas'" style="width:100%;height:100%;"></div>

                 
              </div>

              <div style="font-size: 18px; font-weight: 600;text-align: center; height: calc(100% - 450px);">
                UMAP Projection of Spots ({{props.binSize}})
              </div>
      </div>
    </div>
  </div>
  `
};
