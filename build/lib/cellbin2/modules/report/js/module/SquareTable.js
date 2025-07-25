
const SquareTable = {
  props: {
    moduleTitle: String,
    msg: Array,
    data: Array,
  },
  setup(props) {
    const ifShowExplain = ref(false);
    const colNum = Object.keys(props.data[0]).length - 1;
    const colWidth = ref(`${990 / colNum}px`);
    onMounted(() => {
      console.log('Mount SquareTable: ', props);
    });

    return {
      props,
      ifShowExplain,
      colWidth
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
      <div>
        <template v-for="desc in props.msg">
          <h6 class="collapse-title text-blue">{{desc.title}}</h6>
          <span class="collapse-text">{{desc.content}}</span>
          <br />
        </template>
      </div>
    </div>
    <div class="module-content-box" style="display: flex;justify-content: space-evenly;">
      <el-table :data="props.data" style="width: 100%" :header-cell-style="{'background-color': '#efefef'}">
        <el-table-column prop="binSize" v-if="props.data[0].binSize" label="Bin Size" 
        width="162" align="center" /></el-table-column>
		<el-table-column prop="datatype" v-if="props.data[0].datatype" label="data type" 
        :width="colWidth" align="center" />
        <el-table-column prop="cellCount" v-if="props.data[0].cellCount" label="Cell Count" 
        :width="colWidth" align="center" />
		<el-table-column prop="Fraction_cells" v-if="props.data[0].Fraction_cells" label="Fraction of cells (genes > 200, nucleus )" 
        :width="colWidth" align="center" />
		<el-table-column prop="cell_to_tissue" v-if="props.data[0].cell_to_tissue" label="cell UMI / tissue UMI" 
        :width="colWidth" align="center" />
		</el-table-column>
        <el-table-column label="Mean" align="center">
          <el-table-column prop="meanReads" v-if="props.data[0].meanReads" label="Reads" 
          :width="colWidth" align="center" />
          <el-table-column prop="meanCellArea" v-if="props.data[0].meanCellArea" label="Cell Area" 
          :width="colWidth" align="center" />
          <el-table-column prop="meanGeneType" v-if="props.data[0].meanGeneType" label="Gene Type" :width="colWidth" align="center"></el-table-column>
          <el-table-column prop="meanDNBCount" v-if="props.data[0].meanDNBCount" label="DNB Count" :width="colWidth" align="center"></el-table-column>
          <el-table-column prop="meanMID" v-if="props.data[0].meanMID" label="MID" 
          :width="colWidth" align="center">
		  </el-table-column>
        </el-table-column>
        <el-table-column label="Median" align="center">
          <el-table-column prop="medianReads" v-if="props.data[0].medianReads" label="Reads" :width="colWidth" align="center"></el-table-column>
          <el-table-column prop="medianCellArea" v-if="props.data[0].medianCellArea" label="Cell Area" 
          :width="colWidth" align="center" />
          <el-table-column prop="medianGeneType" v-if="props.data[0].medianGeneType" label="Gene Type" :width="colWidth" align="center"></el-table-column>
          <el-table-column prop="medianDNBCount" v-if="props.data[0].medianDNBCount" label="DNB Count" :width="colWidth" align="center"></el-table-column>
          <el-table-column prop="medianMID" v-if="props.data[0].medianMID" label="MID" align="center">
		  </el-table-column>
        </el-table-column>
      </el-table>
    </div>
  </div>
  `
};