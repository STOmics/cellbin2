
const Information = {
  props: {
    moduleTitle: String,
    msg: Array,
    data: Array
  },
  setup(props) {
    const ifShowExplain = ref(false);
    const layout = {
      "margin": { "l": 70, "t": 30, "r": 20, "b": 50 },
      "showlegend": false,
      "paper_bgcolor": "#FFF3"
    };
    const config = {
      "displayModeBar": false,
      "scrollZoom": false,
      "displaylogo": false
    };
    onMounted(() => {
      console.log('Mount Saturation: ', props);
      props.data.forEach((item, index) => {
        Plotly.newPlot(document.getElementById('saturation_' + index), item.traces, { ...layout, ...item.layout }, { ...config, ...item.config });
      });
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
    <div class="module-content-box" style="display: flex;justify-content: space-evenly;">
      <template v-for="(item, index) in props.data">
        <div :id="'saturation_' + index" style="width:380px;height:400px;"> </div>
      </template>
    </div>
  </div>
  `
};