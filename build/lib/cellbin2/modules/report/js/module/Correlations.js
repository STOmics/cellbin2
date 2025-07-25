
const Correlations = {
  props: {
    moduleTitle: String,
    msg: Array,
    data: Array,
    yAxisTitle: String
  },
  setup(props) {
    const ifShowExplain = ref(false);

    const trace = {
      type: 'heatmap',
      hoverongaps: false,
      hovertemplate: `Protein: %{x}<br>${props.yAxisTitle}: %{y}<br>Corr: %{z}<extra></extra>`,
      colorscale: [
        [0, '#16518B'],
        [0.25, '#7DB6D6'],
        [0.5, '#FFFFFF'],
        [0.75, '#F2A851'],
        [1, '#9D4D09']
      ],
      zmin: -1,
      zmax: 1,
      colorbar: {
        len: 0.5,
        thickness: 18,
        x: 1.05,
        xanchor: "left",
        tickformat: '.1e',
        tickmode: 'array',
        ticktext: ['-1.0', '-0.5', '0', '0.5', '1.0'],
        tickvals: [-1.0, -0.5, 0, 0.5, 1],
        xpad: 0,
        yanchor: "left",
        y: 0.765,
        ticks: 'outside',
        ticklen: 2,
        tickwidth: 2,
        titleside: 'top',
        title: 'Corr',
        titlefont: {
          size: 12,
          weight: 500
        }
      }
    };

    const layout = {
      annotations: [],
      margin: { 'l': 120, 't': 24, 'r': 0, 'b': 80, pad: 0 },
      xaxis: {
        tickvals: props.data.trace.x,
        ticktext: props.data.trace.x,
        tickfont: {
          size: 6
        },
        title: {
          text: 'Protein',
          font: {
            color: '#000000',
            size: 18
          },
          standoff: 1000
        },
        ticks: '',
        side: 'bottom'
      },
      yaxis: {
        tickvals: props.data.trace.y,
        ticktext: props.data.trace.y,
        tickfont: {
          size: 6
        },
        title: {
          text: props.yAxisTitle,
          font: {
            color: '#000000',
            size: 18
          }
        },
        ticks: '',
        ticksuffix: ' ',
        autosize: false
      }
    };

    const config = {
      displaylogo: false,
      modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines", "autoScale2d", "lasso2d", "select2d"]
    };

    onMounted(() => {
      console.log('Mount Correlation: ', props);
      Plotly.newPlot('protein_correlation', [{ ...trace, ...props.data.trace }], layout, config);
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
      <div id="protein_correlation" style="overflow:auto;height: 856px;width: 900px;"></div>
    </div>
  </div>
  `
};