
const MarkerCluster = {
  props: {
    msg: Array,
    data: Array,
    prefix: String
  },
  setup(props) {
    const ifShowExplain = ref(false);
    const geneBoxId = props.prefix + '-geneMarker';
    const proteinBoxId = props.prefix + '-proteinMarker';

    const trace = {
      type: 'heatmap',
      hovertemplate: 'x: %{x}<br>y: %{y}<br>value: %{z}<br>name: %{customdata}<extra></extra>',
      colorscale: [
        [0, '#FFFFFF'],
        [1, '#F2AA56']
      ],
      zmin: 0,
      zmax: 1,
      colorbar: {
        len: 0.7,
        thickness: 18,
        x: 1.05,
        xanchor: "left",
        tickformat: '.1e',
        tickmode: 'array',
        ticktext: ['0', '0.25', '0.5', '0.75', '1.0'],
        tickvals: [0, 0.25, 0.5, 0.75, 1],
        xpad: 0,
        yanchor: "left",
        y: 0.7,
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
      title: {
        font: {
          color: '#000000',
          size: 18
        },
        xanchor: 'center',
        y: 0.98,
        yanchor: 'top'
      },
      yaxis: {
        zeroline: false,
        dtick: 1,
      },
      xaxis: {
        zeroline: false,
        dtick: 1,
      },
      margin: { 'l': 50, 't': 30, 'r': 50, 'b': 120 },
      showlegend: true,
      dragmode: "pan",
      plot_bgcolor: "#FFF3",
      paper_bgcolor: "#FFF3",
    };

    const config = {
      displaylogo: false,

      scrollZoom: false,
      displayModeBar: false,
    };

    onMounted(() => {
      console.log('Mount MarkerCluster: ', props, geneBoxId);
      Plotly.newPlot(geneBoxId, [mergeObjects(trace, props.data.gene.trace)], mergeObjects(layout, props.data.gene.layout), config);
      Plotly.newPlot(proteinBoxId, [mergeObjects(trace, props.data.protein.trace)], mergeObjects(layout, props.data.protein.layout), config);
    });

    return {
      props,
      geneBoxId,
      proteinBoxId,
      ifShowExplain
    };
  },
  template: `
  <div class="module-box" style="width: 1200px; ">
    <div class="module-title-box">
      <div class="title-box-left">
        <span class="title-icon"></span>
        <span class="title-label">Top Markders by Cluster</span>
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
    <div class="module-content-box" style="display: flex; justify-content: center;flex-direction: column;align-items: center;">
      <div :id="geneBoxId" style="  ">
      </div>
      <div :id="proteinBoxId" style=" margin-top: 24px;">
      </div>
    </div>
  </div>
  `
};