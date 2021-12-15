using UnityEngine;
using System.Collections;

public sealed class mfGizmos
{

	public static Color Color = Color.green;
	 
	public static void DrawWireCircle(Vector3 center, float radius, float theta = 0.1f)
	{
		if (radius <= 0) return;

		if (theta < 0.0001f) theta = 0.0001f; 

		// 设置颜色
		Color lastColor = Gizmos.color;
		Gizmos.color = Color;

		// 绘制圆环 
		Vector3 beginPoint = Vector3.zero;
		Vector3 endPoint = Vector3.zero;
		Vector3 firstPoint = Vector3.zero;
		for (float iTheta = 0; iTheta < 2 * Mathf.PI; iTheta += theta)
		{
			float x = radius * Mathf.Cos(iTheta);
			float z = radius * Mathf.Sin(iTheta);
			endPoint.x = x;
			endPoint.z = z; 
			if (iTheta == 0)
			{
				firstPoint = endPoint;
			}
			else
			{
				Gizmos.DrawLine(beginPoint + center, endPoint + center);
			}
			beginPoint = endPoint;
		}

		// 绘制最后一条线段
		Gizmos.DrawLine(firstPoint + center, beginPoint + center);
		// 恢复默认颜色
		Gizmos.color = lastColor;  
	}

	/// <summary>
	/// 根据Cube的8个顶点绘制
	/// </summary>
	/// <param name="leftBottomBack"></param>
	/// <param name="leftBottomFront"></param>
	/// <param name="leftTopBack"></param>
	/// <param name="leftTopFront"></param>
	/// <param name="rightBottomBack"></param>
	/// <param name="rightBottomFront"></param>
	/// <param name="rightTopBack"></param>
	/// <param name="rightTopFront"></param>
	public static void DrawWireCube(Vector3 leftBottomBack, Vector3 leftBottomFront, Vector3 leftTopBack, Vector3 leftTopFront,
		Vector3 rightBottomBack, Vector3 rightBottomFront, Vector3 rightTopBack, Vector3 rightTopFront) 
	{
		Color lastColor = Gizmos.color;
		Gizmos.color = Color; 

		// back side
		Gizmos.DrawLine(leftBottomBack, rightBottomBack);
		Gizmos.DrawLine(leftBottomBack, leftTopBack);
		Gizmos.DrawLine(rightTopBack, rightBottomBack);
		Gizmos.DrawLine(rightTopBack, leftTopBack); 
		// front side
		Gizmos.DrawLine(leftBottomFront, rightBottomFront);
		Gizmos.DrawLine(leftBottomFront, leftTopFront);
		Gizmos.DrawLine(rightTopFront, rightBottomFront);
		Gizmos.DrawLine(rightTopFront, leftTopFront); 
		// left side
		Gizmos.DrawLine(leftBottomBack, leftBottomFront);
		Gizmos.DrawLine(leftTopBack, leftTopFront);   
		// right side
		Gizmos.DrawLine(rightBottomBack, rightBottomFront);
		Gizmos.DrawLine(rightTopBack, rightTopFront);

		Gizmos.color = lastColor;
	}

	/// <summary>
	/// 根据6个面到参照点的偏移绘制，所有offset全是正值,表示各面到参照点的距离 
	/// </summary>
	/// <param name="referPoint"></param> 
	/// <param name="leftOffset"></param>
	/// <param name="rightOffset"></param>
	/// <param name="bottomOffset"></param>
	/// <param name="topOffset"></param>
	/// <param name="backOffset"></param>
	/// <param name="frontOffset"></param>
	public static void DrawWireCube(Vector3 referPoint, float leftOffset, float rightOffset, float bottomOffset, 
		float topOffset, float backOffset, float frontOffset)  
	{
		// 计算6个面到中心的距离
		float leftPos = referPoint.x - leftOffset;
		float rightPos = referPoint.x + rightOffset;
		float bottomPos = referPoint.y - bottomOffset;
		float topPos = referPoint.y + topOffset;
		float backPos = referPoint.z - backOffset;
		float frontPos = referPoint.z + frontOffset; 
		// 计算8个顶点
		Vector3 leftBottomBack = Vector3.zero;
		leftBottomBack.x = leftPos;
		leftBottomBack.y = bottomPos;
		leftBottomBack.z = backPos;
		Vector3 leftBottomFront = Vector3.zero;
		leftBottomFront.x = leftPos;
		leftBottomFront.y = bottomPos;
		leftBottomFront.z = frontPos; 
		Vector3 leftTopBack = Vector3.zero;
		leftTopBack.x = leftPos;
		leftTopBack.y = topPos;
		leftTopBack.z = backPos;
		Vector3 leftTopFront = Vector3.zero;
		leftTopFront.x = leftPos;
		leftTopFront.y = topPos;
		leftTopFront.z = frontPos;
		Vector3 rightBottomBack = Vector3.zero;
		rightBottomBack.x = rightPos;
		rightBottomBack.y = bottomPos;
		rightBottomBack.z = backPos;
		Vector3 rightBottomFront = Vector3.zero;
		rightBottomFront.x = rightPos;
		rightBottomFront.y = bottomPos;
		rightBottomFront.z = frontPos;
		Vector3 rightTopBack = Vector3.zero;
		rightTopBack.x = rightPos;
		rightTopBack.y = topPos;
		rightTopBack.z = backPos;
		Vector3 rightTopFront = Vector3.zero;
		rightTopFront.x = rightPos;
		rightTopFront.y = topPos;
		rightTopFront.z = frontPos;

		DrawWireCube(leftBottomBack, leftBottomFront, leftTopBack, leftTopFront, rightBottomBack, rightBottomFront, rightTopBack, rightTopFront);
	}

	public static void DrawWireCube(Vector3 aabbMin, Vector3 aabbMax)
	{
		// 计算8个顶点
		Vector3 leftBottomBack = Vector3.zero;
		leftBottomBack.x = aabbMin.x;
		leftBottomBack.y = aabbMin.y;
		leftBottomBack.z = aabbMin.z;
		Vector3 leftBottomFront = Vector3.zero;
		leftBottomFront.x = aabbMin.x;
		leftBottomFront.y = aabbMin.y;
		leftBottomFront.z = aabbMax.z;
		Vector3 leftTopBack = Vector3.zero;
		leftTopBack.x = aabbMin.x;
		leftTopBack.y = aabbMax.y;
		leftTopBack.z = aabbMin.z;
		Vector3 leftTopFront = Vector3.zero;
		leftTopFront.x = aabbMin.x;
		leftTopFront.y = aabbMax.y;
		leftTopFront.z = aabbMax.z;
		Vector3 rightBottomBack = Vector3.zero;
		rightBottomBack.x = aabbMax.x;
		rightBottomBack.y = aabbMin.y;
		rightBottomBack.z = aabbMin.z;
		Vector3 rightBottomFront = Vector3.zero;
		rightBottomFront.x = aabbMax.x;
		rightBottomFront.y = aabbMin.y;
		rightBottomFront.z = aabbMax.z;
		Vector3 rightTopBack = Vector3.zero;
		rightTopBack.x = aabbMax.x;
		rightTopBack.y = aabbMax.y;
		rightTopBack.z = aabbMin.z;
		Vector3 rightTopFront = Vector3.zero;
		rightTopFront.x = aabbMax.x;
		rightTopFront.y = aabbMax.y;
		rightTopFront.z = aabbMax.z;

		DrawWireCube(leftBottomBack, leftBottomFront, leftTopBack, leftTopFront, rightBottomBack, rightBottomFront, rightTopBack, rightTopFront);
	} 
	 
}
