
const Title = {
  props: {
    moduleTitle: String,
    moduleName: String,

  },
  setup(props) {
    const ifShowExplain = ref(false);

    onMounted(() => {
      console.log('Mount Title : ', props);
    });

    return {
      props,
      ifShowExplain
    };
  },
  template: `
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
      <el-select v-model="selectedImage" clearable placeholder="Select" style="max-width: 170px;">
        <el-option v-for="item in baseImageList" :key="item.value" :label="item.label" :value="item.value" />
      </el-select>
    </div>

    <div class="module-msg" v-if="ifShowExplain">
      {{ props.msg }}
    </div>
  `
};