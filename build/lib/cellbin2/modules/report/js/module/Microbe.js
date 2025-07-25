
const Microbe = {
  props: {
    moduleTitle: String,
    msg: Array,
    data: Array,
    header: String,
    src: String
  },
  setup(props) {
    const ifShowExplain = ref(false);
    onMounted(() => {
      console.log('Mount Microbe: ', props);
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
    <div class="module-content-box" style="display: flex;justify-content: space-between;">
      <div class="module-content-left" style=" width: 540px">
        <el-table :data="props.data"  style="width: 100%;"   >
          <el-table-column prop="label" :label="props.header" width="250px" align="left"></el-table-column>
          <el-table-column prop="percentage" label="Percentage" align="center"></el-table-column>
          <el-table-column prop="value" label="Count" align="right"></el-table-column>
        </el-table>
      </div>
      <div class="module-content-right" style=" width: 564px;height: 470px;display: flex;justify-content: center;align-items: center; border: solid 1px #f2f2f2;border-radius: 5px;">
        <img :src="props.src" alt="" loading="lazy" style="width: 400px;height: 400px; object-fit: contain;"/>
      </div>
    </div>
  </div>
  `
};