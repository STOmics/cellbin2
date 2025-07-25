
const Counts = {
  props: {
    moduleTitle: String,
    msg: Array,
    data: Array
  },
  setup(props) {
    const ifShowExplain = ref(false);

    const layout = {
      bargap: 0,
      bargroupgap: 0,
      barmode: "overlay",
      xaxis: {
        title: {
          text: "Log10(1+Count)",
          font: {
            color: '#000000',
            size: 18
          }
        }
      },
      yaxis: {
        title: {
          text: "Number of Spots",
          font: {
            color: '#000000',
            size: 18
          }
        }
      },
      margin: { 'l': 80, 't': 30, 'r': 0, 'b': 50, pad: 0 },
    };

    const config = {
      renderer: 'webgl',
      displaylogo: false,
      modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines", "autoScale2d", "lasso2d", "select2d"]
    };

    onMounted(() => {
      console.log('Mount Counts: ', props);
      Plotly.newPlot('histogram_of_protein_counts', props.data.traces, layout, config);
    });

    return {
      props,
      ifShowExplain
    };
  },
  template: `
  <div class="module-box" style="width: 1200px; ">
    <div class="module-title-box">
      <div class="title-box-left">
        <span class="title-icon"></span>
        <span class="title-label">{{ props.moduleTitle }}</span>
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
    <div class="module-content-box" style="display: flex;justify-content: space-evenly;">
      <div id="histogram_of_protein_counts" style="overflow:auto;height: 500px;width: 900px;"></div>
    </div>
  </div>
  `
};