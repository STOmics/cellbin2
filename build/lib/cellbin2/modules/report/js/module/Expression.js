
const Expression = {
  props: {
    containerId: String,
    moduleTitle: String,
    msg: String,
    heatmapImageObj: Object,
    baseImageList: Array
  },
  setup(props) {
    const { heatmapImageObj, baseImageList, containerId } = props;
    const ifShowTissueBinsLayer = ref(true);
    const heatmapImageRef = ref();
    const ifShowExplain = ref(false);
    const heatmapOpacity = ref(50);
    let panzoomInstance;







    function renderSummaryProteinExpression(eleId = 'summary-gene-expression-imagebox') {
      console.log('render expression, id: ', eleId);
      panzoomInstance = Panzoom(document.getElementById(eleId), {
        minScale: 0.5,
        maxScale: 4,
      });
      document.getElementById(eleId).addEventListener('wheel', panzoomInstance.zoomWithWheel);
    };

    function reset() {
      panzoomInstance.reset();
    };

    const selectedImage = ref(baseImageList[0].value);

    const currentColorBar = ref(heatmapImageObj.tissueBinsColorBar);
    const currentHeatmapImageSrc = ref(heatmapImageObj.tissueBins);

    watch(() => ifShowTissueBinsLayer.value, (nv, ov) => {
      if (nv) {
        currentHeatmapImageSrc.value = heatmapImageObj.tissueBins;
        currentColorBar.value = heatmapImageObj.tissueBinsColorBar;
      } else {
        currentHeatmapImageSrc.value = heatmapImageObj.allBins;
        currentColorBar.value = heatmapImageObj.allBinsColorBar;
      };
    });

    const currentBaseImageSrc = computed(() => {
      let targetSrc;
      baseImageList.forEach(item => {
        if (item.value === selectedImage.value) {
          targetSrc = item.src.source;
        };
      });
      return targetSrc;
    });

    const changeHeatmapOpacity = (val) => {
      if (val == 0 || val == 100) {
        heatmapOpacity.value = val;
      };
      heatmapImageRef.value.style.opacity = heatmapOpacity.value / 100;
    };

    onMounted(() => {
      console.log(123, props);
      renderSummaryProteinExpression(containerId);
    });

    return {
      props,
      containerId,
      ifShowTissueBinsLayer,
      selectedImage,
      heatmapImageRef,
      changeHeatmapOpacity,
      reset,
      heatmapOpacity,
      ifShowExplain,
      baseImageList,
      currentHeatmapImageSrc,
      currentBaseImageSrc,
      currentColorBar
    };
  },
  template: `
  <div class="module-box" style="width: 780px;margin-right: 24px;">
  
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
      <el-select v-model="selectedImage" placeholder="Select" style="max-width: 170px;">
        <el-option v-for="item in baseImageList" :key="item.value" :label="item.label" :value="item.value" />
      </el-select>
    </div>

    <div class="module-msg" v-if="ifShowExplain">
      <div v-for="item in props.msg">
      {{ item }}
      </div> 
    </div>

    <div class="module-content-box" style="display: flex; width: 100%;height: 660px;">
      <div class="panzoom-box" style="width: 0;flex:1; height: 100%; background-color: #000000;position: relative;border-radius: 5px;">
        <el-button @click="reset()" circle  plain size="small" style="position: absolute; right: 5px; top: 5px; z-index: 99; opacity: 0.7;visibility: hidden;"  >
          <el-icon :size="18"><RefreshLeft /></el-icon>
        </el-button>
        <div :id="containerId" style="position: relative;width: 100%;height: 100%;">
          <img v-if="currentBaseImageSrc" :src="currentBaseImageSrc" alt="" style="width: 100%;height: 100%;position:absolute;opacity: 1;object-fit: contain;">
          <img v-if="currentHeatmapImageSrc" ref="heatmapImageRef" :src="currentHeatmapImageSrc" alt="" style="width: 100%;height: 100%;position:absolute;opacity: 0.5;object-fit: contain;">
        </div>

      </div>
      <div style="height: 100% ; width: 50px; position: relative;">
        <img id="summary-rna-colorbar" style="position: absolute;bottom: 0;" :src="currentColorBar" alt="">
      </div>
    </div>

    <div class="module-footer-box" style="display: flex; justify-content: space-between;  padding: 18px 8px 0 8px;">
      <el-switch v-model="ifShowTissueBinsLayer" class="mb-2"
        style="--el-switch-on-color: #B321F7; --el-switch-off-color: #8539ff" active-text="Bins under tissue"
        inactive-text="All bins">
      </el-switch>
      <div style="display: flex;width: 380px;">
        <el-button @click="changeHeatmapOpacity(0)" style="margin: 0 10px;">Image</el-button>
        <el-slider v-model="heatmapOpacity" :min='0' :max=100 @input=changeHeatmapOpacity() :format-tooltip="formatTooltip" ></el-slider>
        <el-button @click="changeHeatmapOpacity(100)" style="margin: 0 10px;">Expression</el-button>
      </div>
    </div>

  </div>
  `
};