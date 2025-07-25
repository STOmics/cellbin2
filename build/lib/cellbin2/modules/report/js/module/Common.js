
const Common = {
 
  props: {
    withMarginRight: Boolean,
    moduleTitle: String,
    msg: String,
    data: Array
  },
  setup(props) {
    const ifShowExplain = ref(false);
    onMounted(() => {
      console.log('Common Module mounted, props: ', props);

    });

    return {
      props,
      ifShowExplain,
    };
  },
  template: `
  <div :class="{'module-box': true, 'with-margin-right': props.withMarginRight}" style="width: 588px;">
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
      <div>
        <template v-for="desc in props.msg">
          <h6 class="collapse-title text-blue">{{desc.title}}</h6>
          <span class="collapse-text">{{desc.content}}</span>
          <br />
        </template>
      </div>
    </div>

    <div class="module-content-box">
      <el-table :data="props.data"  style="width: 100%;" :show-header=false  >
        <el-table-column prop="label" width="320px" align="left"></el-table-column>
        <el-table-column prop="value" align="right"></el-table-column>
      </el-table>
    </div>
  </div>
  `
};