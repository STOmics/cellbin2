const Cbdistribution = {
  props: {
    moduleTitle: String,
    msg: Array,
    data: Array
  },
  setup(props) {
    const ifShowExplain = ref(false);
    const toolTip = document.getElementById('hover-layer2');
    const hideLayer = () => {
      toolTip.style.display = 'none';
      toolTip.removeEventListener('mouseleave', hideLayer);
    };
    onMounted(() => {
      console.log('Mount Distribution: ', props);
    });

    function showTip(event, info) {
      console.log('---- mousemove', event.offsetX, event.offsetY);
      if (event.offsetX < 10 || event.offsetX > 312 || event.offsetY < 10 || event.offsetY > 360) {
        toolTip.style.display = 'none';
        return
      };
      toolTip.style.display = 'block';
      toolTip.style.left = event.pageX + 'px';
      toolTip.style.top = event.pageY + 'px';
      toolTip.innerHTML = `
    <div> <span class="left_label">Max</span> <span>: ${info.max}</span></div>
    <div> <span class="left_label">Q1</span> <span>: ${info.q1}</span></div>
    <div> <span class="left_label">Q2</span> <span>: ${info.q2}</span></div>
    <div> <span class="left_label">Q3</span> <span>: ${info.q3}</span></div>
    <div> <span class="left_label">Min</span> <span>: ${info.min}</span></div>
    `;
      toolTip.addEventListener('mouseleave', hideLayer);
    };
    return {
      props,
      ifShowExplain,
      showTip
    };
  },
  template: `
  <div class="module-box" style="width: 1200px; ">
    <div class="module-title-box">
      <div class="title-box-left">
        <span class="title-icon"></span>
        <span class="title-label">CellBin Distribution</span>
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
    <div class="module-content-box" style="display: flex;justify-content: space-evenly;padding-top:24px;">
      <template v-for="(item, index) in props.data">
        <img style="width:312px; object-fit: contain;" :src="item.src" @mousemove="e => showTip(e, item.info)"/>
      </template>
    </div>
  </div>
  `
};