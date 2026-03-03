
const TrackPoint = {
  props: {
    containerId: String,
    withMarginRight: Boolean,
    moduleTitle: String,
    src: String,
  },
  setup(props) {
    const ifShowExplain = ref(false);
    let panzoomInstance;
    function renderTissuesegPlot(eleId = 'tissuesegImg') {
      panzoomInstance = Panzoom(document.getElementById('tissuesegImg'), {
        minScale: 0.5,
        maxScale: 2,
      });
      document.getElementById('tissuesegImg').addEventListener('wheel', panzoomInstance.zoomWithWheel);
    }

    function reset() {
      panzoomInstance.reset();
    };
    onMounted(() => {
      console.log('Tissueseg  mounted, props: ', props);
      renderTissuesegPlot(props.containerId);
    });

    return {
      props,
      ifShowExplain,
      reset
    };
  },
  template: `
  <div :class="{'module-box': true}" style="width: 280px;height: 280px;">
    <div class="module-title-box" >
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
    <div class="module-content-box">
      <div class="tissue_segment_plot panzoom-box" :id="containerId" style="position: relative;width: 250px;height: 250px;background-color: #000000;">
        <el-button @click="reset()" circle plain size="small" style="position: absolute; right: 5px; top: 5px; z-index: 99; opacity: 0.7;visibility: hidden;"  >
          <el-icon :size="18"><RefreshLeft /></el-icon>
        </el-button>
        <img id="tissuesegImg" :src="props.src" style="width: 100%;height: 100%;position:absolute;object-fit: contain;"> 
<!--        <img id="tissuesegImg" :src="props.src" style="width: 100%;height: 100%;position:absolute;object-fit: contain;"> -->
<!--        <img id="tissuesegImg" :src="props.src" style="width: 100%;height: 100%;position:absolute;object-fit: contain;"> -->
<!--        <img id="tissuesegImg" :src="props.src" style="width: 100%;height: 100%;position:absolute;object-fit: contain;"> -->
<!--        <img id="tissuesegImg" :src="props.src" style="width: 100%;height: 100%;position:absolute;object-fit: contain;"> -->
      </div>
    </div>
  </div>
  `
};
